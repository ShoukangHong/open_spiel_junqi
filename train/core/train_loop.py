"""Shared AlphaZero training loop — game-agnostic.

Usage (game-specific wrapper):
    from train.core.train_loop import run_training

    run_training(
        game_name="othello",
        config_class=OthelloTrainConfig,
        build_model_fn=build_othello_model,
        play_game_fn=play_game,
        SymmetryClass=OthelloSymmetry,
    )
"""

import logging
import multiprocessing as mp
import os
import queue
import time
from dataclasses import asdict

# Linux: CUDA requires spawn (fork is default on Linux but incompatible with GPU)
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method("spawn")

# Pin BLAS to single-thread before numpy/torch import (actors re-import this module)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pyspiel
import torch
torch.set_num_threads(1)

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
from train.core.types import Losses, TrainInput
from train.core.weak_move import (
    accum_wstats, nn_raw_after_move, reset_wstats, try_weak_move,
    wstats_summary)

_LATEST = -999


# ── WDL value target helpers ────────────────────────────────────────────────

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


# ── Actor process (multi-actor mode) ────────────────────────────────────────

def actor_process(config_class, cfg_dict, incoming_q, result_q, state_queue,
                  actor_id, play_game_fn, shm_name=None):
    """Actor subprocess: self-play loop pushing states to the trainer."""
    try:
        import os as _os
        allowed = sorted(_os.sched_getaffinity(0))
        if len(allowed) >= 4:
            num_gpus = max(cfg_dict.get('num_gpus', 1), 1)
            leave = min(num_gpus, len(allowed) - 1)
            _os.sched_setaffinity(0, set(allowed[:-leave]))
    except Exception as _e:
        print(f"[actor-{actor_id}] CPU affinity failed: {_e}", flush=True)
    cfg = config_class(**cfg_dict)
    game = pyspiel.load_game(cfg.game)
    actor_shm = None
    if shm_name is not None:
        obs_flat = int(np.prod(game.observation_tensor_shape()))
        act_flat = game.num_distinct_actions()
        from train.batch_mcts.shm import ActorShm
        actor_shm = ActorShm(shm_name, cfg.mcts_batch_size,
                             obs_flat, act_flat, act_flat, create=False)
    ev_main = SharedEvaluator(game, incoming_q, result_q, actor_id,
                              model_id="main", actor_shm=actor_shm)
    ev_best = SharedEvaluator(game, incoming_q, result_q, actor_id,
                              model_id="best", actor_shm=actor_shm)
    mcts_cfg = MCTSConfig(
        max_simulations=cfg.max_simulations, batch_size=cfg.mcts_batch_size,
        uct_c=cfg.uct_c, policy_epsilon=cfg.policy_epsilon,
        policy_alpha=cfg.policy_alpha,
        draw_penalty=cfg.draw_penalty, verbose=False)
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
            if cfg.best_model_prob > 0:
                use_best = rng.random() < cfg.best_model_prob
        mcts_p0 = mcts_main
        mcts_p1 = (mcts_best if use_best else mcts_main)
        if use_best and rng.random() < 0.5:
            mcts_p0, mcts_p1 = mcts_p1, mcts_p0
        states_info, returns, rare_games, wstats = play_game_fn(
            game, mcts_p0, mcts_p1, cfg, rng, logger=logger,
            init_state=init_state, allow_weak=allow_weak)
        for rs in rare_games:
            pending.append((rs, False, "rare", use_best))
        if (tag_override == "rare" and init_state is not None
                and not init_state.is_terminal()):
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


# ── Main training loop ──────────────────────────────────────────────────────

def run_training(
        game_name: str,
        config_class,
        build_model_fn,
        play_game_fn,
        SymmetryClass,
        *,
        config_path: str = None,
        output_path: str = None,
        fresh: bool = False,
):
    """Run a complete AlphaZero training run.

    Args:
        game_name: pyspiel game name (e.g. "othello", "xiangqi").
        config_class: TrainConfig subclass for this game.
        build_model_fn: callable(game, cfg) → Model.
        play_game_fn: self-play function with signature
            (game, mcts_black, mcts_white, config, rng, *, logger,
             init_state, allow_weak) → (states_info, returns, rare_games, wstats).
        SymmetryClass: symmetry-augmentation class (or None for no symmetry).
        config_path: optional path to JSON config file.
        fresh: if True, start from scratch ignoring existing checkpoints.
    """

    # ── Config & logging ───────────────────────────────────────────────────
    cfg = setup_config_and_logging(config_path, config_class, fresh,
                                   output_dir=output_path)

    game, model, buffer, sym, n_updates, start_step, mcts_config = \
        init_training(cfg, None, build_model_fn, ReplayBuffer, SymmetryClass)
    _log = logging.info
    samples_per_step = max(
        int(cfg.replay_buffer_size * cfg.buffer_sampling_frac),
        cfg.train_batch_size)

    # ── Inference servers (one per GPU) ─────────────────────────────────
    num_gpus = max(getattr(cfg, 'num_gpus', 1), 1)
    servers = []  # list of (server, a_start, a_end, gpu_id)
    actors = []
    if cfg.num_actors > 1:
        model.save_checkpoint(_LATEST)
        cfg_dict = asdict(cfg)
        cfg_dict["path"] = cfg.path
        obs_flat = int(np.prod(game.observation_tensor_shape()))
        mask_flat = game.num_distinct_actions()

        # Best-model state dict (shared across all servers)
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

        actors_per_gpu = (cfg.num_actors + num_gpus - 1) // num_gpus
        for gpu_id in range(num_gpus):
            server = InferenceServer(build_model_fn=build_model_fn, gpu_id=gpu_id)
            a_start = gpu_id * actors_per_gpu
            a_end = min(a_start + actors_per_gpu, cfg.num_actors)
            if a_start >= a_end:
                break  # fewer actors than GPUs

            # Register actor slots for this GPU (local IDs: 0, 1, ...)
            for local_id in range(a_end - a_start):
                server.register_actor(local_id, cfg.mcts_batch_size,
                                      obs_flat, mask_flat, mask_flat)
            server.register_model("main", model._model.state_dict(),
                                  cfg.nn_width, cfg.nn_depth)
            server.register_model("best", best_sd, cfg.nn_width, cfg.nn_depth)
            server.start(cfg.game, cfg.inference_batch_size)
            servers.append((server, a_start, a_end, gpu_id))

        # Spawn actors, each connected to its GPU's server
        for server, a_start, a_end, gpu_id in servers:
            incoming_q = server.incoming_queue
            for global_id in range(a_start, a_end):
                local_id = global_id - a_start
                result_q = server.result_queue(local_id)
                shm_name = server.actor_shm_name(local_id)
                state_q = mp.Queue(maxsize=200)
                p = mp.Process(target=actor_process,
                               args=(config_class, cfg_dict, incoming_q, result_q,
                                     state_q, local_id, play_game_fn, shm_name),
                               name=f"actor-gpu{gpu_id}-{local_id}")
                p.start()
                actors.append((p, state_q))
        _log(f"[train] {len(actors)} actors on {len(servers)} GPU(s)"
             f" ({actors_per_gpu} each)")

    # ── Training state ─────────────────────────────────────────────────────
    global_rng = np.random.RandomState(cfg.seed + start_step)
    _last_eval_time = -cfg.eval_min_interval
    _eval_thread = None

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
                mcts_best = mcts_main
                best_file = os.path.join(cfg.path, "best_step.txt")
                if os.path.exists(best_file):
                    try:
                        with open(best_file) as f:
                            bs = int(f.read().strip())
                        ckpt = os.path.join(cfg.path, f"checkpoint-{bs}.pt")
                        if os.path.exists(ckpt):
                            best_model = build_model_fn(game, cfg)
                            best_model.load_checkpoint(bs)
                            ev_best = PyTorchEvaluator(game, best_model)
                            mcts_best = BatchMCTS(
                                game, mcts_config, ev_best,
                                random_state=np.random.RandomState(
                                    cfg.seed + step * 1000 + 1))
                    except Exception:
                        pass
                game_logger = GameLogger(cfg.path, 0)
                pending = []

                while total_states < samples_per_step:
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
                        mb, mw = mw, mb
                    states_info, returns, rare_games, wstats = play_game_fn(
                        game, mb, mw, cfg, global_rng, logger=game_logger,
                        init_state=init_state, allow_weak=allow_weak)

                    for rs in rare_games:
                        pending.append((rs, False, "rare", use_best))

                    if (tag_override == "rare" and init_state is not None
                            and not init_state.is_terminal()
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
                # Multi-actor path
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
                p = batch.policy
                p = np.where(p > 0, p, 1.0)
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

            # ── Broadcast weights to all GPU servers ──────────────────
            if cfg.num_actors > 1 and servers:
                model.save_checkpoint(_LATEST)
                main_sd = model._model.state_dict()
                for server, _, _, _ in servers:
                    server.update_weights("main", main_sd,
                                          cfg.inference_batch_size)
                _last_best = getattr(run_training, "_last_best_step", 0)
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
                                for server, _, _, _ in servers:
                                    server.update_weights(
                                        "best", best_sd, cfg.inference_batch_size)
                                _log(f"  [best] updated → step {cur_best}")
                            run_training._last_best_step = cur_best
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
        for p, q in actors:
            if p.is_alive():
                p.terminate()
        for p, q in actors:
            p.join(timeout=5)
        for server, _, _, _ in servers:
            server.stop()
            server.shutdown()
            server.terminate()
        for child in mp.active_children():
            child.terminate()
            child.join(timeout=3)
        _log("[train] Shutdown complete.")
