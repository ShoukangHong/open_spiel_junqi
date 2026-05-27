"""Tests for train/core/* — game-agnostic utilities."""

import json
import os
import tempfile

import numpy as np

from train.core.checkpoint import find_latest_checkpoint
from train.core.base_config import BaseTrainConfig, load_base_config
from train.core.weak_move import accum_wstats, reset_wstats, wstats_summary


# ── checkpoint ──────────────────────────────────────────────────────────────

def test_find_latest_empty():
    with tempfile.TemporaryDirectory() as d:
        assert find_latest_checkpoint(d) == 0


def test_find_latest_single():
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "checkpoint-5.pt"), "w") as f:
            f.write("x")
        assert find_latest_checkpoint(d) == 5


def test_find_latest_multiple():
    with tempfile.TemporaryDirectory() as d:
        for s in [3, 15, 8, 42]:
            with open(os.path.join(d, f"checkpoint-{s}.pt"), "w") as f:
                f.write("x")
        assert find_latest_checkpoint(d) == 42


def test_find_latest_ignores_other_files():
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "checkpoint-10.pt"), "w") as f:
            f.write("x")
        with open(os.path.join(d, "train.log"), "w") as f:
            f.write("x")
        with open(os.path.join(d, "checkpoint-abc.pt"), "w") as f:
            f.write("x")
        assert find_latest_checkpoint(d) == 10


# ── config ──────────────────────────────────────────────────────────────────

def test_load_base_config_defaults():
    cfg = BaseTrainConfig()
    assert cfg.learning_rate == 3e-4
    assert cfg.max_steps == 300


def test_load_base_config_from_json():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "cfg.json")
        with open(p, "w") as f:
            json.dump({"learning_rate": 0.0005, "max_steps": 50}, f)
        cfg = load_base_config(p, BaseTrainConfig)
        assert cfg.learning_rate == 0.0005
        assert cfg.max_steps == 50


def test_load_base_config_missing_file():
    cfg = load_base_config("nonexistent.json", BaseTrainConfig)
    assert cfg.learning_rate == 3e-4  # defaults


# ── weak-move stats ─────────────────────────────────────────────────────────

def test_wstats_accum():
    cfg = BaseTrainConfig()
    accum_wstats(cfg, {"rare": 1, "weak": 3})
    accum_wstats(cfg, {"rare": 0, "weak": 1, "weak_final": 2})
    s = wstats_summary(cfg)
    assert "rare=1" in s
    assert "weak=4" in s


def test_wstats_reset():
    cfg = BaseTrainConfig()
    accum_wstats(cfg, {"rare": 5})
    reset_wstats(cfg)
    assert not hasattr(cfg, "_wstats_total")


def test_wstats_empty():
    cfg = BaseTrainConfig()
    assert wstats_summary(cfg) == "weak=(none)"


# ── backward compat ─────────────────────────────────────────────────────────

def test_backward_compat_imports():
    """Old names should still be importable from train.train_othello."""
    from train.train_othello import (
        TrainConfig, ReplayBuffer, GameLogger, play_game,
        _nn_raw_after_move, _try_weak_move,
        _accum_wstats, _wstats_summary, _reset_wstats,
    )
    assert TrainConfig is not None
    assert ReplayBuffer is not None
    assert GameLogger is not None
    assert play_game is not None
    assert _try_weak_move is not None


# ── main test runner ───────────────────────────────────────────────────────

def main():
    tests = [
        test_find_latest_empty, test_find_latest_single,
        test_find_latest_multiple, test_find_latest_ignores_other_files,
        test_load_base_config_defaults, test_load_base_config_from_json,
        test_load_base_config_missing_file,
        test_wstats_accum, test_wstats_reset, test_wstats_empty,
        test_backward_compat_imports,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  {fn.__name__}: PASSED")
        except Exception as e:
            print(f"  {fn.__name__}: FAILED — {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    print(f"\n{'=' * 40}")
    print(f"{'ALL PASSED' if failed == 0 else f'{failed} FAILED'}")
    print("=" * 40)
    assert failed == 0


def test_core():
    main()


if __name__ == "__main__":
    test_core()
