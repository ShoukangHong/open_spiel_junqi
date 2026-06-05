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




def _mcts_wdl(q_value, draw_rate):
    """Reconstruct WDL from MCTS Q and draw rate.  Returns [w, d, l]."""
    w = max(( q_value + 1.0 - draw_rate) / 2.0, 0.0)
    l = max((-q_value + 1.0 - draw_rate) / 2.0, 0.0)
    s = w + draw_rate + l
    if s > 0:
        return np.array([w / s, draw_rate / s, l / s], dtype=np.float32)
    return np.array([0.0, 1.0, 0.0], dtype=np.float32)


def _outcome_wdl(ret_scalar):
    """Game outcome scalar → one-hot WDL."""
    if ret_scalar > 0.1:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if ret_scalar < -0.1:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return np.array([0.0, 1.0, 0.0], dtype=np.float32)


def _mixed_target(game_ret, q_value, draw_rate, alpha):
    """Mixed WDL: alpha * game_outcome + (1-alpha) * MCTS WDL."""
    mcts = _mcts_wdl(q_value, draw_rate)
    game = _outcome_wdl(game_ret)
    return alpha * game + (1.0 - alpha) * mcts


from train.core.model_builder import build_othello_model as build_model


def actor_process(cfg_dict, incoming_q, result_q, state_queue, actor_id=0):
    # Leave last core for inference server
    try:
        import os as _os
        allowed = sorted(_os.sched_getaffinity(0))
        if len(allowed) >= 4:
            _os.sched_setaffinity(0, set(allowed[:-1]))
    except Exception as _e:
        print(f"[actor-{actor_id}] CPU affinity failed: {_e}", flush=True)
    cfg = OthelloTrainConfig(**cfg_dict)
    game = pyspiel.load_game(cfg.game)
    ev_main = SharedEvaluator(game, incoming_q, result_q, actor_id,
                              model_id="main")
    ev_best = SharedEvaluator(game, incoming_q, result_q, actor_id,
                              model_id="best")
    mcts_cfg = MCTSConfig(
        max_simulations=cfg.max_simulations, batch_size=cfg.mcts_batch_size,
        uct_c=cfg.uct_c, policy_epsilon=cfg.policy_epsilon,
        policy_alpha=cfg.policy_alpha, verbose=False)
    mcts_main = BatchMCTS(game, mcts_cfg, ev_main,
                          random_state=np.random.RandomState())
    mcts_best = BatchMCTS(game, mcts_cfg, ev_best,
                          random_state=np.random.RandomState())
    rng = np.random.RandomState()
    logger = GameLogger(cfg.path, actor_id)
    pending = []
    while True:
        if pending:
            init_state, allow_weak, tag_override, use_best = pending.pop()
        else:
            init_state, allow_weak, tag_override, use_best = None, True, "", False
            # Randomly pick opponent model for this game
            if cfg.best_model_prob > 0:
                use_best = rng.random() < cfg.best_model_prob
        mcts_p0 = mcts_main
        mcts_p1 = (mcts_best if use_best else mcts_main)
        if use_best and rng.random() < 0.5:
            mcts_p0, mcts_p1 = mcts_p1, mcts_p0  # alternate colors
        states_info, returns, rare_games, wstats = play_game(
            game, mcts_p0, mcts_p1, cfg, rng, logger=logger,
            init_state=init_state, allow_weak=allow_weak)
        for rs in rare_games:
            pending.append((rs, False, "rare", use_best))
        if tag_override == "rare" and init_state is not None:
            weak_player = 1 - init_state.current_player()
            if returns[weak_player] > 0:
                tag_override = "rare_flip"
                wstats["rare_flip"] = wstats.get("rare_flip", 0) + 1
        if tag_override:
            for i in range(len(states_info)):
                item = states_info[i]
                states_info[i] = (item[0], item[1], item[2], item[3],
                                  tag_override, *item[5:])
        rare_at = len(init_state.history()) if init_state is not None else 0
        try:
            state_queue.put((states_info, returns, wstats, rare_at), timeout=1)
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
            cfg.nn_width, cfg.nn_depth)
        # Load best model if recorded, else copy main as initial best
        best_sd = model._model.state_dict()
        best_file = os.path.join(cfg.path, "best_step.txt")
        if os.path.exists(best_file):
            try:
                with open(best_file) as f:
                    best_step = int(f.read().strip())
                ckpt = os.path.join(cfg.path, f"checkpoint-{best_step}.pt")
                if os.path.exists(ckpt):
                    best_sd = torch.load(
                        ckpt, map_location="cpu",
                        weights_only=False)["model_state_dict"]
                    _log(f"[train] Best model: step {best_step}")
            except Exception:
                pass
        inference_server.register_model(
            "best", best_sd, cfg.nn_width, cfg.nn_depth)
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
                ev_main = PyTorchEvaluator(game, model)
                mcts_main = BatchMCTS(game, mcts_config, ev_main,
                                      random_state=np.random.RandomState(
                                          cfg.seed + step * 1000))
                # Load best model if available
                mcts_best = mcts_main  # fallback: same as main
                best_file = os.path.join(cfg.path, "best_step.txt")
                if os.path.exists(best_file):
                    try:
                        with open(best_file) as f:
                            bs = int(f.read().strip())
                        ckpt = os.path.join(cfg.path, f"checkpoint-{bs}.pt")
                        if os.path.exists(ckpt):
                            best_model = build_othello_model(game, cfg)
                            best_model.load_checkpoint(bs)
                            ev_best = PyTorchEvaluator(game, best_model)
                            mcts_best = BatchMCTS(
                                game, mcts_config, ev_best,
                                random_state=np.random.RandomState(
                                    cfg.seed + step * 1000 + 1))
                    except Exception:
                        pass
                game_logger = GameLogger(cfg.path, 0)
                pending = []  # (init_state, allow_weak, tag_override)

                while total_states < samples_per_step:
                    # ── Decide what game to play next ──────────────────
                    if pending:
                        (init_state, allow_weak,
                         tag_override, use_best) = pending.pop()
                    else:
                        init_state, allow_weak, tag_override = None, True, ""
                        use_best = (cfg.best_model_prob > 0
                                    and global_rng.random()
                                    < cfg.best_model_prob)

                    mb = mcts_main
                    mw = mcts_best if use_best else mcts_main
                    if use_best and global_rng.random() < 0.5:
                        mb, mw = mw, mb  # alternate colors
                    states_info, returns, rare_games, wstats = play_game(
                        game, mb, mw, cfg, global_rng, logger=game_logger,
                        init_state=init_state, allow_weak=allow_weak)

                    # Enqueue rare states for future games
                    for rs in rare_games:
                        pending.append((rs, False, "rare", use_best))

                    # Detect rare-flip: weak-move player won the fork game
                    if (tag_override == "rare" and init_state is not None
                            and returns[1 - init_state.current_player()] > 0):
                        tag_override = "rare_flip"
                        wstats["rare_flip"] = wstats.get("rare_flip", 0) + 1
                    accum_wstats(cfg, wstats)

                    game_outcome_p0 = returns[0]
                    game_length = len(states_info)
                    rare_at = 0
                    if init_state is not None:
                        rare_at = len(init_state.history())
                    offset = rare_at
                    denom = max(offset + game_length - 1, 1)
                    for i, item in enumerate(states_info):
                        obs, mask, policy, cur_player = item[:4]
                        tag = item[4] if len(item) > 4 else ""
                        q_value = item[5] if len(item) > 5 else 0.0
                        draw_rate = item[6] if len(item) > 6 else 0.0
                        if tag_override:
                            tag = tag_override
                        alpha = (offset + i) / denom
                        val = _mixed_target(returns[cur_player], q_value,
                                            draw_rate, alpha)
                        buffer.append(obs, mask, policy, val, tag, step=step)

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
                            states_info, returns, wstats, rare_at = q.get_nowait()
                        except queue.Empty:
                            continue
                        accum_wstats(cfg, wstats)
                        game_outcome_p0 = returns[0]

                        game_length = len(states_info)
                        offset = rare_at
                        denom = max(offset + game_length - 1, 1)
                        for i, item in enumerate(states_info):
                            obs, mask, policy, cur_player = item[:4]
                            tag = item[4] if len(item) > 4 else ""
                            q_value = item[5] if len(item) > 5 else 0.0
                            draw_rate = item[6] if len(item) > 6 else 0.0
                            alpha = (offset + i) / denom
                            val = _mixed_target(returns[cur_player], q_value,
                                                draw_rate, alpha)
                            buffer.append(obs, mask, policy, val, tag,
                                          step=step)

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
                        batch.value)
                    batch = TrainInput(observation=obs, legals_mask=mask,
                                       policy=policy, value=value)
                model._entropy_weight = cfg.entropy_weight
                loss = model.update(batch)
                losses_list.append(loss)
                # Policy entropy (nats) — per-sample, then averaged
                p = batch.policy
                p = np.where(p > 0, p, 1.0)  # log(1)=0, neutral for entropy
                sample_ent = -np.sum(p * np.log(p), axis=-1)
                entropies.append(float(np.mean(sample_ent)))

            if losses_list:
                n = len(losses_list)
                avg_loss = Losses(
                    policy=sum(l.policy for l in losses_list) / n,
                    value=sum(l.value for l in losses_list) / n,
                    l2=sum(l.l2 for l in losses_list) / n,
                    v_kl=sum(l.v_kl for l in losses_list) / n,
                    top1=sum(l.top1 for l in losses_list) / n,
                    p_kl=sum(l.p_kl for l in losses_list) / n,
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
                _log(f"  [checkpoint] Saved {ckpt_path}")

            # Broadcast latest weights to actors + server
            if cfg.num_actors > 1:
                model.save_checkpoint(_LATEST)
                inference_server.update_weights(
                    "main", model._model.state_dict(),
                    cfg.inference_batch_size)
                # Refresh best model if best_step.txt changed
                _last_best = getattr(main, "_last_best_step", 0)
                best_file = os.path.join(cfg.path, "best_step.txt")
                if os.path.exists(best_file):
                    try:
                        with open(best_file) as f:
                            cur_best = int(f.read().strip())
                        if cur_best != _last_best:
                            ckpt = os.path.join(
                                cfg.path, f"checkpoint-{cur_best}.pt")
                            if os.path.exists(ckpt):
                                best_sd = torch.load(
                                    ckpt, map_location="cpu",
                                    weights_only=False)["model_state_dict"]
                                inference_server.update_weights(
                                    "best", best_sd, cfg.inference_batch_size)
                                _log(f"  [best] updated → step {cur_best}")
                            main._last_best_step = cur_best
                    except Exception:
                        pass

            # ── Evaluation ─────────────────────────────────────────────
            _eval_thread, _last_eval_time = maybe_trigger_eval(
                step, cfg, _eval_thread, _last_eval_time,
                run_eval_background)

        # Final save
        model.save_checkpoint(cfg.max_steps)
        _log(f"\n[train] Done. Final checkpoint: {cfg.max_steps}")

    finally:
        _log("[train] Shutting down ...")
        buffer.flush()
        buffer.close()
        # 1. Kill actors first (before server, to avoid deadlock on GPU queue)
        for p, q in actors:
            if p.is_alive():
                p.terminate()
        for p, q in actors:
            p.join(timeout=5)
        # 2. Kill server
        inference_server.terminate()
        # 3. Belt-and-suspenders: any leftover children
        for child in mp.active_children():
            child.terminate()
            child.join(timeout=3)
        _log("[train] Shutdown complete.")

if __name__ == "__main__":
    main()
