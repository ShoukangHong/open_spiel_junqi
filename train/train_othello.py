r"""Othello AlphaZero training with BatchMCTS + PyTorch.

Usage:
    python train_othello.py                  # use defaults
    python train_othello.py --fresh           # start from scratch
    python train_othello.py --config my.json  # custom config

Architecture:
  - Single-process synchronous self-play + training
  - BatchMCTS for efficient search (batch_size=32 leaf eval)
  - PyTorch ResNet model
  - Simple replay buffer (numpy ring buffer)
  - Regular checkpointing and evaluation
"""

import argparse
import json
import multiprocessing as mp
import os
import queue

# Linux: CUDA requires spawn (fork is default on Linux but incompatible with GPU)
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method("spawn")
import logging
import sys
import threading
import time
from dataclasses import asdict, dataclass
from typing import Optional

# Make sure the project root is on sys.path so that `train.*` imports work
# regardless of where the script is invoked from.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pyspiel

from train.batch_mcts.config import MCTSConfig
from train.batch_mcts.evaluator import PyTorchEvaluator
from train.batch_mcts.mcts import BatchMCTS
from train.batch_mcts.shared_evaluator import InferenceServer, SharedEvaluator
from train.core.base_config import BaseTrainConfig
from train.core.checkpoint import find_latest_checkpoint
from train.core.game_logger import GameLogger
from train.core.replay_buffer import ReplayBuffer
from train.core.train_utils import (
    init_training, maybe_trigger_eval, run_eval_background,
    setup_config_and_logging)
from train.core.weak_move import (
    accum_wstats, nn_raw_after_move, reset_wstats, try_weak_move,
    wstats_summary)
from train.games.othello.config import OthelloTrainConfig
from train.games.othello.play import play_game

TrainConfig = OthelloTrainConfig  # backward compat
_nn_raw_after_move = nn_raw_after_move    # backward compat
_try_weak_move = try_weak_move            # backward compat
_accum_wstats = accum_wstats              # backward compat
_wstats_summary = wstats_summary          # backward compat
_reset_wstats = reset_wstats              # backward compat
from train.model.othello_resnet import Losses, Model, OthelloResNet, TrainInput
from train.model.symmetry import OthelloSymmetry

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))




def _make_value(ret, value_classes):
    if value_classes == 1:
        return float(ret)
    r = float(ret)
    if r > 0.1:      return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    elif r < -0.1:   return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:            return np.array([0.0, 1.0, 0.0], dtype=np.float32)


from train.core.model_builder import build_othello_model as build_model


def actor_process(cfg_dict, incoming_q, result_q, state_queue, actor_id=0):
    cfg = OthelloTrainConfig(**cfg_dict)
    game = pyspiel.load_game(cfg.game)
    evaluator = SharedEvaluator(game, incoming_q, result_q, actor_id)
    mcts_cfg = MCTSConfig(
        max_simulations=cfg.max_simulations, batch_size=cfg.mcts_batch_size,
        uct_c=cfg.uct_c, policy_epsilon=cfg.policy_epsilon,
        policy_alpha=cfg.policy_alpha,
        value_classes=cfg.value_classes, verbose=False)
    mcts = BatchMCTS(game, mcts_cfg, evaluator,
                     random_state=np.random.RandomState())
    rng = np.random.RandomState()
    logger = GameLogger(cfg.path, actor_id)
    pending = []
    while True:
        if pending:
            init_state, allow_weak, tag_override = pending.pop()
        else:
            init_state, allow_weak, tag_override = None, True, ""
        states_info, returns, rare_games, wstats = play_game(
            game, mcts, cfg, rng, logger=logger,
            init_state=init_state, allow_weak=allow_weak)
        for rs in rare_games:
            pending.append((rs, False, "rare"))
        if tag_override:
            for i in range(len(states_info)):
                obs, mask, policy, cp, _tag = states_info[i]
                states_info[i] = (obs, mask, policy, cp, tag_override)
        try:
            state_queue.put((states_info, returns, wstats), timeout=1)
        except queue.Full:
            print("[actor] WARNING: queue full, dropping game", flush=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Othello AlphaZero training (BatchMCTS + PyTorch)")
    parser.add_argument("--config", default="train_othello_config.json")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    cfg = setup_config_and_logging(
        args.config, OthelloTrainConfig, args.fresh)

    game, model, buffer, sym, n_updates, start_step, mcts_config = \
        init_training(cfg, None, build_model, ReplayBuffer, OthelloSymmetry)
    _log = logging.info
    samples_per_step = max(
        int(cfg.replay_buffer_size * cfg.buffer_sampling_frac),
        cfg.train_batch_size)

    _LATEST = -999

    # ── Shared inference server (multi-actor GPU batching) ───────────────
    inference_server = InferenceServer()
    actors = []
    if cfg.num_actors > 1:
        model.save_checkpoint(_LATEST)
        cfg_dict = asdict(cfg)
        cfg_dict["path"] = cfg.path
        incoming_q = inference_server.incoming_queue
        for i in range(cfg.num_actors):
            inference_server.register_actor(i)
        # Start server AFTER all actors are registered
        inference_server.register_model(
            "main", model._model.state_dict(),
            cfg.nn_width, cfg.nn_depth, cfg.value_classes)
        inference_server.start(cfg.game, cfg.inference_batch_size)
        for i in range(cfg.num_actors):
            result_q = inference_server.result_queue(i)
            state_q = mp.Queue(maxsize=200)
            p = mp.Process(target=actor_process,
                           args=(cfg_dict, incoming_q, result_q,
                                 state_q, i),
                           name=f"actor-{i}")
            p.start()
            actors.append((p, state_q))
        _log(f"[train] Spawned {cfg.num_actors} actor processes")

    # Training state
    global_rng = np.random.RandomState(cfg.seed + start_step)
    _last_eval_time = -cfg.eval_min_interval  # force first eval immediately
    _eval_thread = None     # track running eval

    try:
        for step in range(start_step + 1, cfg.max_steps + 1):
            t0 = time.time()

            # ── Self-play ──────────────────────────────────────────────
            total_states = 0
            total_games = 0
            outcomes = {"p0": 0, "p1": 0, "draw": 0}

            if cfg.num_actors == 1:
                # Single-process path
                evaluator = PyTorchEvaluator(game, model,
                                             value_classes=cfg.value_classes)
                mcts = BatchMCTS(game, mcts_config, evaluator,
                                 random_state=np.random.RandomState(
                                     cfg.seed + step * 1000))
                game_logger = GameLogger(cfg.path, 0)
                pending = []  # (init_state, allow_weak, tag_override)

                while total_states < samples_per_step:
                    # ── Decide what game to play next ──────────────────
                    if pending:
                        init_state, allow_weak, tag_override = pending.pop()
                    else:
                        init_state, allow_weak, tag_override = None, True, ""

                    states_info, returns, rare_games, wstats = play_game(
                        game, mcts, cfg, global_rng, logger=game_logger,
                        init_state=init_state, allow_weak=allow_weak)
                    accum_wstats(cfg, wstats)

                    # Enqueue rare states for future games
                    for rs in rare_games:
                        pending.append((rs, False, "rare"))

                    game_outcome_p0 = returns[0]
                    for item in states_info:
                        obs, mask, policy, cur_player = item[:4]
                        tag = item[4] if len(item) > 4 else ""
                        if tag_override:
                            tag = tag_override
                        val = _make_value(returns[cur_player], cfg.value_classes)
                        buffer.append(obs, mask, policy, val, tag)

                    if game_outcome_p0 > 0:
                        outcomes["p0"] += 1
                    elif game_outcome_p0 < 0:
                        outcomes["p1"] += 1
                    else:
                        outcomes["draw"] += 1

                    total_states += len(states_info)
                    total_games += 1


            else:
                # Multi-actor path: collect from subprocess queues
                while total_states < samples_per_step:
                    for _, q in actors:
                        try:
                            states_info, returns, wstats = q.get_nowait()
                        except queue.Empty:
                            continue
                        accum_wstats(cfg, wstats)
                        game_outcome_p0 = returns[0]

                        for item in states_info:
                            obs, mask, policy, cur_player = item[:4]
                            tag = item[4] if len(item) > 4 else ""
                            val = _make_value(returns[cur_player], cfg.value_classes)
                            buffer.append(obs, mask, policy, val, tag)

                        if game_outcome_p0 > 0:
                            outcomes["p0"] += 1
                        elif game_outcome_p0 < 0:
                            outcomes["p1"] += 1
                        else:
                            outcomes["draw"] += 1

                        total_states += len(states_info)
                        total_games += 1
                        if total_states >= samples_per_step:
                            break
                    time.sleep(0.001)

            selfplay_time = time.time() - t0

            # ── Training ───────────────────────────────────────────────
            train_t0 = time.time()
            losses_list = []
            entropies = []

            for _ in range(n_updates):
                batch = buffer.sample(cfg.train_batch_size)
                if sym is not None:
                    obs, mask, policy, value = sym.augment_batch(
                        batch.observation, batch.legals_mask, batch.policy,
                        batch.value, cfg.value_classes)
                    batch = TrainInput(observation=obs, legals_mask=mask,
                                       policy=policy, value=value)
                loss = model.update(batch)
                losses_list.append(loss)
                # Policy entropy (nats) — per-sample, then averaged
                p = batch.policy
                p = np.where(p > 0, p, 1.0)  # log(1)=0, neutral for entropy
                sample_ent = -np.sum(p * np.log(p), axis=-1)
                entropies.append(float(np.mean(sample_ent)))

            if losses_list:
                avg_loss = Losses(
                    policy=sum(l.policy for l in losses_list) / len(losses_list),
                    value=sum(l.value for l in losses_list) / len(losses_list),
                    l2=sum(l.l2 for l in losses_list) / len(losses_list),
                )
                avg_entropy = sum(entropies) / len(entropies)
            else:
                avg_loss = None
                avg_entropy = 0.0
            train_time = time.time() - train_t0

            elapsed = time.time() - t0
            states_per_s = total_states / max(selfplay_time, 0.001)

            # ── Logging ────────────────────────────────────────────────
            log_line = (
                f"[step {step:3d}/{cfg.max_steps}] "
                f"games={total_games:3d}  states={total_states:4d}  "
                f"buffer={len(buffer):5d}/{buffer.total_seen:5d}"
                f" unique={buffer.unique_states}"
                f"  tags={buffer.tag_counts()}"
                f"  weak={wstats_summary(cfg)}  "
                f"states/s={states_per_s:.1f}  "
                f"selfplay={selfplay_time:.1f}s  train={train_time:.1f}s"
            )
            if avg_loss is not None:
                log_line += (f"\n               loss={avg_loss}"
                             f"  entropy={avg_entropy:.3f}  |")
            g = total_games or 1
            log_line += (f"  p0={outcomes['p0']/g:.1%}"
                         f"  p1={outcomes['p1']/g:.1%}"
                         f"  draw={outcomes['draw']/g:.1%}")
            _log(log_line)
            reset_wstats(cfg)

            # ── Checkpoint ─────────────────────────────────────────────
            if step % cfg.checkpoint_freq == 0:
                ckpt_path = model.save_checkpoint(step)
                buffer.save(os.path.join(
                    cfg.path, f"buffer-checkpoint-{step}.npz"))
                _log(f"  [checkpoint] Saved {ckpt_path}")

            # Broadcast latest weights to actors + server
            if cfg.num_actors > 1:
                model.save_checkpoint(_LATEST)
                inference_server.update_weights(
                    "main", model._model.state_dict(),
                    cfg.inference_batch_size)

            # ── Evaluation ─────────────────────────────────────────────
            _eval_thread, _last_eval_time = maybe_trigger_eval(
                step, cfg, _eval_thread, _last_eval_time,
                run_eval_background)

        # Final save
        model.save_checkpoint(cfg.max_steps)
        _log(f"\n[train] Done. Final checkpoint: {cfg.max_steps}")

    finally:
        inference_server.terminate()
        for p, q in actors:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)

if __name__ == "__main__":
    main()
