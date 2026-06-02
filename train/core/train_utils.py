"""Training-loop utilities — game-agnostic (config, resume, eval trigger)."""

import json
import logging
import os
import sys
import threading
import time

from train.core.checkpoint import find_latest_checkpoint


def setup_config_and_logging(config_path: str, config_class,
                             fresh: bool, output_dir: str = None):
    """Load config, setup logging, handle resume reload.

    Returns (cfg, output_dir).
    """
    cfg = config_class()
    if config_path:
        cfg = _load_config(config_path, config_class)

    if output_dir is None:
        output_dir = cfg.path
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_file),
                  logging.StreamHandler(sys.stdout)])

    # Resume: reload from stored config in output dir
    saved_config = os.path.join(output_dir, "train_config.json")
    if not fresh and os.path.exists(saved_config):
        cfg = _load_config(saved_config, config_class)
        logging.info(f"[train] Reloaded config from {saved_config}")

    # Save merged config for future resumes
    with open(os.path.join(output_dir, "train_config.json"), "w") as f:
        json.dump({k: v for k, v in cfg.__dict__.items()
                   if not k.startswith("_")}, f, indent=2, default=str)

    logging.info(f"[train] game={cfg.game}  nn_width={cfg.nn_width}"
                 f"  nn_depth={cfg.nn_depth}"
                 f"  value_head=WDL(3)")
    logging.info(f"[train] max_sim={cfg.max_simulations}"
                 f"  mcts_batch={cfg.mcts_batch_size}"
                 f"  infer_batch={cfg.inference_batch_size}"
                 f"  buffer={cfg.replay_buffer_size}")
    logging.info(f"[train] max_steps={cfg.max_steps}"
                 f"  ckpt_freq={cfg.checkpoint_freq}")
    logging.info(f"[train] path={output_dir}")

    return cfg


def init_training(cfg, game_module, model_builder, ReplayBuffer_class,
                  Symmetry_class):
    """Create game, model, buffer, symmetry; try resume from checkpoint.

    Returns (game, model, buffer, sym, n_updates, start_step, mcts_config).
    """
    import numpy as np
    import pyspiel

    from train.batch_mcts.config import MCTSConfig

    game = pyspiel.load_game(cfg.game)
    obs_shape = game.observation_tensor_shape()
    logging.info(f"[train] obs_shape={obs_shape}"
                 f"  num_actions={game.num_distinct_actions()}")

    model = model_builder(game, cfg)
    logging.info(f"[train] Model params: {model.num_trainable_variables}"
                 f"  lr={cfg.learning_rate:.0e}")

    buffer = ReplayBuffer_class(max_size=cfg.replay_buffer_size)
    samples_per_step = max(
        int(cfg.replay_buffer_size * cfg.buffer_sampling_frac),
        cfg.train_batch_size)
    n_updates = samples_per_step * cfg.symmetry // cfg.train_batch_size
    sym = Symmetry_class() if cfg.symmetry > 1 else None
    logging.info(f"[train] buffer_sampling_frac={cfg.buffer_sampling_frac}"
                 f"  symmetry={cfg.symmetry}  n_updates={n_updates}")

    mcts_config = MCTSConfig(
        max_simulations=cfg.max_simulations,
        batch_size=cfg.mcts_batch_size,
        uct_c=cfg.uct_c,
        policy_epsilon=cfg.policy_epsilon,
        policy_alpha=cfg.policy_alpha,
        verbose=False,
    )

    # Resume
    start_step = find_latest_checkpoint(cfg.path)
    if start_step > 0:
        model.load_checkpoint(start_step)
        buf_path = os.path.join(cfg.path,
                                f"buffer-checkpoint-{start_step}.npz")
        if os.path.exists(buf_path):
            buffer.load(buf_path)
            logging.info(f"[train] Resumed from checkpoint-{start_step}"
                         f" (buffer: {len(buffer)} states)")
        else:
            logging.info(f"[train] Resumed from checkpoint-{start_step}"
                         f" (buffer file not found, starting empty)")
    else:
        logging.info("[train] No checkpoint found, starting fresh")

    return (game, model, buffer, sym, n_updates, start_step, mcts_config)


def maybe_trigger_eval(step, cfg, _eval_thread, _last_eval_time,
                       eval_func):
    """If conditions met, start a background eval thread.

    Returns updated (_eval_thread, _last_eval_time).
    """
    if step % cfg.checkpoint_freq != 0 or step <= 0:
        return _eval_thread, _last_eval_time

    now = time.time()
    eval_idle = _eval_thread is None or not _eval_thread.is_alive()
    if not eval_idle or now - _last_eval_time < cfg.eval_min_interval:
        return _eval_thread, _last_eval_time

    _last_eval_time = now

    ckpts = []
    for f in os.listdir(cfg.path):
        if not (f.startswith("checkpoint-") and f.endswith(".pt")):
            continue
        try:
            s = int(f.split("-")[1].split(".")[0])
            if s < step:
                ckpts.append(s)
        except (ValueError, IndexError):
            pass
    ckpts.sort(reverse=True)
    refs = ckpts[:cfg.eval_reference_count]
    # Pad with at most one random when fewer checkpoints exist
    if len(refs) < cfg.eval_reference_count and -1 not in refs:
        refs.append(-1)

    _eval_thread = threading.Thread(
        target=eval_func,
        args=(cfg.path, step, refs, cfg.evaluation_window),
        daemon=True)
    _eval_thread.start()
    return _eval_thread, _last_eval_time


def run_eval_background(cfg_path, current_step, ref_steps, num_games):
    """Run model-vs-model eval against multiple references (background)."""
    from train.eval_match import run_match

    mcts_cfg = {"strategy": "mcts",
                "checkpoint_dir": cfg_path,
                "mcts_simulations": 128, "mcts_batch_size": 8, "mcts_uct_c": 1.41}
    cur = dict(mcts_cfg, checkpoint_step=current_step)

    for ref_step in ref_steps:
        if ref_step < 0:
            ref = {"strategy": "random"}
            ref_name = "random"
        else:
            ckpt = os.path.join(cfg_path, f"checkpoint-{ref_step}.pt")
            if os.path.exists(ckpt):
                ref = dict(mcts_cfg, checkpoint_step=ref_step)
                ref_name = f"step{ref_step}"
            else:
                ref = {"strategy": "random"}
                ref_name = "random"

        try:
            score, _ = run_match(cur, ref, num_games=num_games,
                                 temperature=0.1, temp_drop=4, quiet=True)
            keys = [k for k in score if k != "draw"]
            wr = score[keys[0]] / max(num_games, 1)
            logging.info(f"    [eval] step{current_step} vs {ref_name}:  "
                         f"W={score[keys[0]]} L={score[keys[1]]}"
                         f" D={score['draw']} WR={wr:.1%}")
        except Exception as e:
            logging.error(f"    [eval] step{current_step} vs {ref_name}: "
                          f"FAILED — {e}")


def _load_config(path, config_class):
    import json
    cfg = config_class()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        print(f"[config] Loaded {path}")
    else:
        print(f"[config] {path} not found, using defaults")
    return cfg
