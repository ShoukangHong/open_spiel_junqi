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
        handlers=[logging.FileHandler(log_file, encoding="utf-8"),
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

    buffer = ReplayBuffer_class(max_size=cfg.replay_buffer_size,
                                 db_path=os.path.join(cfg.path, "buffer.db"),
                                 recent_db_rows=10000,
                                 game_name=getattr(cfg, 'game', ''))
    sym = Symmetry_class() if cfg.symmetry > 1 else None
    logging.info(f"[train] buffer_sampling_frac={cfg.buffer_sampling_frac}"
                 f"  symmetry={cfg.symmetry}")

    mcts_config = MCTSConfig(
        max_simulations=cfg.max_simulations,
        batch_size=cfg.mcts_batch_size,
        uct_c=cfg.uct_c,
        policy_epsilon=cfg.policy_epsilon,
        policy_alpha=cfg.policy_alpha,
        draw_penalty=cfg.draw_penalty,
        repeat_penalty=cfg.repeat_penalty,
        verbose=False,
    )

    # Resume
    start_step = find_latest_checkpoint(cfg.path)
    if start_step > 0:
        model.load_checkpoint(start_step)
        buffer.rollback(start_step)  # discard states past this checkpoint
        logging.info(f"[train] Resumed from checkpoint-{start_step}"
                     f" (buffer: {len(buffer)} states)")
    else:
        logging.info("[train] No checkpoint found, starting fresh")

    return (game, model, buffer, sym, start_step, mcts_config)


def select_eval_references(ckpts, step, output_dir, ref_count):
    """Build a diverse set of reference checkpoints for evaluation.

    Priority order:
      1. Historical best model (1 slot, from best_step.txt)
      2. Evenly-spaced milestones: (ref_count-1) equal segments across
         [0, step), nearest checkpoint per segment boundary
      3. Most recent unmatched checkpoints
      4. At most one random opponent

    Returns list of step numbers (-1 = random).
    """
    ckpts = sorted([s for s in ckpts if s < step], reverse=True)
    refs = []

    # 1. Best model — auto-init if missing
    best_file = os.path.join(output_dir, "best_step.txt")
    if not os.path.exists(best_file) and ckpts:
        with open(best_file, "w") as f:
            f.write(str(ckpts[-1]))
    if os.path.exists(best_file):
        try:
            with open(best_file) as f:
                best = int(f.read().strip())
            if best > 0 and best < step:
                refs.append(best)
        except (ValueError, OSError):
            pass

    # 2. Milestones: evenly divide [0, step) into (ref_count-1) segments,
    #    pick the nearest checkpoint to each boundary
    n_segments = max(ref_count - 1, 1)
    for i in range(1, n_segments + 1):
        if len(refs) >= ref_count:
            break
        target = step * i // (n_segments + 1)
        pick = None
        best_dist = step
        for s in ckpts:
            if s in refs:
                continue
            d = abs(s - target)
            if d < best_dist or (d == best_dist and (pick is None or s > pick)):
                best_dist = d
                pick = s
        if pick is not None:
            refs.append(pick)

    # 3. Fill remaining slots with most recent unmatched checkpoints
    for s in ckpts:
        if len(refs) >= ref_count:
            break
        if s in refs:
            continue
        refs.append(s)

    # 4. Pad with at most one random
    if len(refs) < ref_count and -1 not in refs:
        refs.append(-1)

    return refs


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
    refs = select_eval_references(ckpts, step, cfg.path,
                                   cfg.eval_reference_count)
    logging.info(f"    [eval] dir={cfg.path}  ckpts_found={len(ckpts)}  "
                 f"refs={refs}")

    _eval_thread = threading.Thread(
        target=eval_func,
        args=(cfg.path, step, refs, cfg.evaluation_window,
              getattr(cfg, 'eval_num_actors', 10),
              cfg.temperature, cfg.temperature_drop),
        daemon=True)
    _eval_thread.start()
    return _eval_thread, _last_eval_time


def run_eval_background(cfg_path, current_step, ref_steps, num_games,
                        num_actors=10, temperature=None, temp_drop=None):
    """Run model-vs-model eval against multiple references (background)."""
    import platform
    if platform.system() == "Windows":
        from train.eval_match import run_match as _run
        _kwargs = {}
    else:
        from train.eval_match import run_match_parallel as _run
        _kwargs = {"num_actors": num_actors}

    # Read training config to match MCTS settings
    import json
    train_cfg_path = os.path.join(cfg_path, "train_config.json")
    tc = {}
    if os.path.exists(train_cfg_path):
        with open(train_cfg_path) as f:
            tc = json.load(f)
    if temperature is None:
        temperature = tc.get("temperature", 0.1)
    if temp_drop is None:
        temp_drop = tc.get("temperature_drop", 25)
    mcts_cfg = {"strategy": "mcts",
                "checkpoint_dir": cfg_path,
                "mcts_simulations": tc.get("max_simulations", 128),
                "mcts_batch_size": tc.get("mcts_batch_size", 8),
                "mcts_uct_c": tc.get("uct_c", 1.41)}
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
                ref_step = -1  # prevent best-model update on fallback

        try:
            score, _ = _run(cur, ref, num_games=num_games,
                            temperature=temperature, temp_drop=temp_drop,
                            quiet=True, **_kwargs)
            keys = [k for k in score if k != "draw"]
            wr = (score[keys[0]] + 0.5 * score.get("draw", 0)) / max(num_games, 1)
            logging.info(f"    [eval] step{current_step} vs {ref_name}:  "
                         f"W={score[keys[0]]} L={score[keys[1]]}"
                         f" D={score['draw']} WR={wr:.1%}")
        except Exception as e:
            logging.error(f"    [eval] step{current_step} vs {ref_name}: "
                          f"FAILED — {e}")
            continue

        # Immediately update best if current beats old best
        best_file = os.path.join(cfg_path, "best_step.txt")
        if os.path.exists(best_file):
            try:
                with open(best_file) as f:
                    old_best = int(f.read().strip())
                if ref_step == old_best and wr >= 0.55:
                    with open(best_file, "w") as f:
                        f.write(str(current_step))
                    logging.info(f"    [eval] → new best model: step{current_step} "
                                 f"(WR={wr:.1%} vs old step{old_best})")
            except (ValueError, OSError):
                pass


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
