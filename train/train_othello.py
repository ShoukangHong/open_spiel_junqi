"""Othello AlphaZero training — click to run, config-driven, resume-capable.

Usage:
    python train_othello.py                  # uses train_othello_config.json
    python train_othello.py --fresh           # ignore existing checkpoints
    python train_othello.py --config my.json  # use a different config file

The script wraps open_spiel's alpha_zero.alpha_zero() with:
  - JSON config file (no CLI flags needed for defaults)
  - Auto-detect latest checkpoint and resume training
  - Sensible overnight CPU defaults for Othello
"""

import argparse
import itertools
import json
import os
import sys

from open_spiel.python.algorithms.alpha_zero import alpha_zero as az_module
from open_spiel.python.utils import spawn

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CONFIG = "train_othello_config.json"


def find_latest_checkpoint(path: str) -> int:
    """Scan *path* for checkpoint-N directories and return the highest N > 0."""
    if not os.path.isdir(path):
        return 0
    best = 0
    for name in os.listdir(path):
        if name.startswith("checkpoint-"):
            step_str = name.split("-", 1)[1]
            if step_str.lstrip("-").isdigit():
                step = int(step_str)
                if step > best:
                    best = step
    return best


def load_config(path: str) -> dict:
    """Return merged dict of built-in defaults + JSON file overrides."""
    defaults = {
        "game": "othello",
        "nn_model": "resnet",
        "nn_width": 32,
        "nn_depth": 5,
        "nn_api": "nnx",
        "max_simulations": 200,
        "actors": 8,
        "evaluators": 1,
        "replay_buffer_size": 8192,
        "replay_buffer_reuse": 4,
        "train_batch_size": 128,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "decouple_weight_decay": False,
        "max_steps": 300,
        "checkpoint_freq": 25,
        "uct_c": 1.41,
        "policy_epsilon": 0.25,
        "policy_alpha": 1.0,
        "temperature": 1.0,
        "temperature_drop": 2,
        "evaluation_window": 50,
        "eval_levels": 7,
        "quiet": False,
        "verbose": False,
        "path": "C:\\Users\\shouk\\othello_train",
        "observation_shape": None,
        "output_size": None,
    }

    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            file_cfg = json.load(f)
        defaults.update(file_cfg)
        print(f"[config] Loaded {path}")
    else:
        print(f"[config] {path} not found, using built-in defaults")

    return defaults


def main():
    parser = argparse.ArgumentParser(description="Othello AlphaZero training")
    parser.add_argument(
        "--config", default=_DEFAULT_CONFIG,
        help="Path to JSON config file",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore existing checkpoints and start a fresh run",
    )
    args = parser.parse_args()

    # Resolve config path
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(_SCRIPT_DIR, config_path)

    cfg = load_config(config_path)

    # Resolve checkpoint path
    out_path = cfg["path"]
    if not os.path.isabs(out_path):
        out_path = os.path.join(_SCRIPT_DIR, out_path)
    cfg["path"] = out_path

    # ── Resume logic ────────────────────────────────────────────────────
    resume_step = 0 if args.fresh else find_latest_checkpoint(out_path)

    if resume_step > 0:
        if resume_step >= cfg["max_steps"]:
            print(
                f"[resume] checkpoint-{resume_step} already reached"
                f" max_steps={cfg['max_steps']}. Increase max_steps in config"
                f" to continue training."
            )
            sys.exit(0)

        print(
            f"[resume] Found checkpoint-{resume_step}, will resume from"
            f" step {resume_step + 1}"
        )

        _original_init_model = az_module._init_model_from_config

        def _init_with_resume(config):
            model = _original_init_model(config)
            model.load_checkpoint(resume_step)
            print(f"[resume] Loaded checkpoint-{resume_step}")
            return model

        az_module._init_model_from_config = _init_with_resume

        # Patch itertools.count so learner's step counter starts from
        # resume_step + 1 instead of 1.  Also affects actor game-numbering
        # (harmless: just starts logs from a higher number).
        _real_count = itertools.count

        def _patched_count(start=0, step=1):
            if start == 1:          # learner + actor both call count(1)
                return _real_count(resume_step + 1, step)
            return _real_count(start, step)

        az_module.itertools.count = _patched_count
    else:
        print("[resume] No checkpoint found, starting fresh training")

    # ── Build Config & launch ───────────────────────────────────────────
    config = az_module.Config(
        game=cfg["game"],
        path=cfg["path"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        decouple_weight_decay=cfg["decouple_weight_decay"],
        train_batch_size=cfg["train_batch_size"],
        replay_buffer_size=cfg["replay_buffer_size"],
        replay_buffer_reuse=cfg["replay_buffer_reuse"],
        max_steps=cfg["max_steps"],
        checkpoint_freq=cfg["checkpoint_freq"],
        actors=cfg["actors"],
        evaluators=cfg["evaluators"],
        uct_c=cfg["uct_c"],
        max_simulations=cfg["max_simulations"],
        policy_alpha=cfg["policy_alpha"],
        policy_epsilon=cfg["policy_epsilon"],
        temperature=cfg["temperature"],
        temperature_drop=cfg["temperature_drop"],
        evaluation_window=cfg["evaluation_window"],
        eval_levels=cfg["eval_levels"],
        nn_model=cfg["nn_model"],
        nn_width=cfg["nn_width"],
        nn_depth=cfg["nn_depth"],
        observation_shape=cfg["observation_shape"],
        output_size=cfg["output_size"],
        quiet=cfg["quiet"],
        verbose=cfg["verbose"],
        nn_api_version=cfg["nn_api"],
    )

    print(f"[train] game={config.game}  model={config.nn_model}"
          f"(w={config.nn_width}, d={config.nn_depth})")
    print(f"[train] max_sim={config.max_simulations}  actors={config.actors}"
          f"  buffer={config.replay_buffer_size}")
    print(f"[train] max_steps={config.max_steps}"
          f"  ckpt_freq={config.checkpoint_freq}")
    print(f"[train] path={config.path}")

    az_module.alpha_zero(config)


if __name__ == "__main__":
    with spawn.main_handler():
        main()
