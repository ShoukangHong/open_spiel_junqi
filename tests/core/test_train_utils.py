"""Tests for train/core/train_utils.py — config, eval refs, wstats, compat."""

import json
import os
import tempfile

import numpy as np

from train.core.base_config import BaseTrainConfig, load_base_config
from train.core.weak_move import accum_wstats, reset_wstats, wstats_summary
from train.core.train_utils import select_eval_references


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
    assert cfg.learning_rate == 3e-4


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


# ── import paths (post-refactor) ──────────────────────────────────────────

def test_import_paths():
    """Verify core modules are importable from their canonical locations."""
    from train.core.replay_buffer import ReplayBuffer
    from train.core.game_logger import GameLogger
    from train.core.weak_move import nn_raw_after_move, try_weak_move
    from train.core.weak_move import accum_wstats, wstats_summary, reset_wstats
    from train.games.othello.config import OthelloTrainConfig
    from train.games.othello.play import play_game
    assert OthelloTrainConfig is not None
    assert ReplayBuffer is not None
    assert GameLogger is not None
    assert play_game is not None
    assert try_weak_move is not None
    assert nn_raw_after_move is not None
    assert accum_wstats is not None
    assert wstats_summary is not None
    assert reset_wstats is not None


# ── eval references ─────────────────────────────────────────────────────────

# Tests updated for historical-best milestone selection.
# select_eval_references now reads all lines from best_step.txt
# (last = current best, rest = history for milestones).

def test_eval_refs_empty():
    refs = select_eval_references([], 10, ".", ref_count=3)
    assert refs == [-1]


def test_eval_refs_one_ckpt():
    refs = select_eval_references([5], 10, ".", ref_count=3)
    assert 5 in refs
    assert -1 in refs
    assert len(refs) == 2


def test_eval_refs_historical_bests():
    ckpts = [10, 20, 35, 50, 65]
    best_file = os.path.join(".", "best_step.txt")
    with open(best_file, "w") as f:
        for s in [10, 20, 35, 50, 65]:
            f.write(f"{s}\n")
    try:
        refs = select_eval_references(ckpts, 70, ".", ref_count=3)
        assert refs[0] == 65      # current best
        assert 20 in refs
        assert 35 in refs
        assert len(refs) == 3
    finally:
        os.remove(best_file)


def test_eval_refs_best_not_duplicated():
    ckpts = [50, 40, 30, 20, 10]
    best_file = os.path.join(".", "best_step.txt")
    with open(best_file, "w") as f:
        for s in [20, 50]:
            f.write(f"{s}\n")
    try:
        refs = select_eval_references(ckpts, 55, ".", ref_count=3)
        assert refs[0] == 50
        assert refs[1] == 20
        assert refs[2] == -1
        assert len(refs) == 3
    finally:
        os.remove(best_file)


def test_eval_refs_no_best_file():
    best_file = os.path.join(".", "best_step.txt")
    if os.path.exists(best_file):
        os.remove(best_file)
    try:
        refs = select_eval_references([30, 20, 10], 35, ".", ref_count=3)
        assert 10 in refs
        assert os.path.exists(best_file)
    finally:
        if os.path.exists(best_file):
            os.remove(best_file)


def test_eval_refs_history_spread():
    ckpts = list(range(20, 201, 20))
    best_file = os.path.join(".", "best_step.txt")
    with open(best_file, "w") as f:
        for s in [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]:
            f.write(f"{s}\n")
    try:
        refs = select_eval_references(ckpts, 210, ".", ref_count=5)
        assert refs[0] == 200
        assert len(refs) >= 4
        for r in refs[1:]:
            assert r in [20, 40, 60, 80, 100, 120, 140, 160, 180]
    finally:
        os.remove(best_file)


def test_equal_spacing_ref_count_4():
    ckpts = list(range(10, 100, 10))
    best_file = os.path.join(".", "best_step.txt")
    with open(best_file, "w") as f:
        for s in [10, 20, 35, 50, 70, 90]:
            f.write(f"{s}\n")
    try:
        refs = select_eval_references(ckpts, 95, ".", ref_count=4)
        assert refs[0] == 90
        assert refs[1] == 20
        assert refs[2] == 35
        assert refs[3] == 50
        assert len(refs) == 4
    finally:
        os.remove(best_file)


def test_equal_spacing_early_training():
    ckpts = [5, 10]
    best_file = os.path.join(".", "best_step.txt")
    with open(best_file, "w") as f:
        for s in [5, 10]:
            f.write(f"{s}\n")
    try:
        refs = select_eval_references(ckpts, 15, ".", ref_count=2)
        assert 10 in refs and 5 in refs
        assert len(refs) == 2
    finally:
        os.remove(best_file)


def test_equal_spacing_ref_count_1():
    ckpts = list(range(10, 51, 10))
    refs = select_eval_references(ckpts, 55, ".", ref_count=1)
    assert len(refs) == 1


def test_eval_refs_count_never_exceeds():
    ckpts = list(range(10, 201, 10))
    for n in [1, 2, 3, 5, 10]:
        best_file = os.path.join(".", "best_step.txt")
        if os.path.exists(best_file):
            os.remove(best_file)
        refs = select_eval_references(ckpts, 205, ".", ref_count=n)
        assert len(refs) <= n
    if os.path.exists(best_file):
        os.remove(best_file)


def test_eval_refs_history_too_short_pads_random():
    ckpts = [50, 40, 30, 20, 10]
    best_file = os.path.join(".", "best_step.txt")
    with open(best_file, "w") as f:
        for s in [20, 50]:
            f.write(f"{s}\n")
    try:
        refs = select_eval_references(ckpts, 55, ".", ref_count=3)
        assert refs[0] == 50
        assert 20 in refs
        assert -1 in refs
        assert len(refs) == 3
    finally:
        os.remove(best_file)


def main():
    tests = [
        test_load_base_config_defaults, test_load_base_config_from_json,
        test_load_base_config_missing_file,
        test_wstats_accum, test_wstats_reset, test_wstats_empty,
        test_backward_compat_imports,
        test_eval_refs_empty, test_eval_refs_one_ckpt,
        test_eval_refs_historical_bests, test_eval_refs_best_not_duplicated,
        test_eval_refs_no_best_file, test_eval_refs_history_spread,
        test_equal_spacing_ref_count_4, test_equal_spacing_early_training,
        test_equal_spacing_ref_count_1,
        test_eval_refs_count_never_exceeds,
        test_eval_refs_history_too_short_pads_random,
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


if __name__ == "__main__":
    main()
