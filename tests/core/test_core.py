"""Tests for train/core/* — game-agnostic utilities."""

import json
import os
import tempfile

import numpy as np

from train.batch_mcts.node import Node
from train.core.checkpoint import find_latest_checkpoint
from train.core.base_config import BaseTrainConfig, load_base_config
from train.core.play import _stable_qdr, assign_players
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
    assert cfg.random_opponent_prob == 0.2
    assert cfg.best_model_prob == 0.3


def test_load_base_config_from_json():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "cfg.json")
        with open(p, "w") as f:
            json.dump({"learning_rate": 0.0005, "max_steps": 50,
                       "random_opponent_prob": 0.5}, f)
        cfg = load_base_config(p, BaseTrainConfig)
        assert cfg.learning_rate == 0.0005
        assert cfg.max_steps == 50
        assert cfg.random_opponent_prob == 0.5


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

# ── select_eval_references ───────────────────────────────────────────────────

from train.core.train_utils import select_eval_references


def test_eval_refs_empty():
    """No checkpoints → just random."""
    refs = select_eval_references([], 10, ".", ref_count=3)
    assert refs == [-1]


def test_eval_refs_one_ckpt():
    """One checkpoint → auto-best + random."""
    refs = select_eval_references([5], 10, ".", ref_count=3)
    assert 5 in refs
    assert -1 in refs
    assert len(refs) == 2


def test_eval_refs_evenly_spaced():
    """ckpts=[10..60], step=65, ref_count=3 → best + 2 evenly-spaced."""
    import os
    best_file = os.path.join(".", "best_step.txt")
    if os.path.exists(best_file):
        os.remove(best_file)
    ckpts = list(range(10, 61, 10))
    refs = select_eval_references(ckpts, 65, ".", ref_count=3)
    assert refs[0] == 10   # best
    # targets: 65//3≈21 → 20, 65*2//3≈43 → 40
    assert 20 in refs
    assert 40 in refs
    assert len(refs) == 3
    os.remove(best_file)


def test_eval_refs_best_not_duplicated():
    """When best == a milestone candidate, don't duplicate."""
    ckpts = [50, 40, 30, 20, 10]
    import os
    best_file = os.path.join(".", "best_step.txt")
    with open(best_file, "w") as f:
        f.write("50")
    try:
        refs = select_eval_references(ckpts, 55, ".", ref_count=3)
        assert refs[0] == 50
        assert 50 not in refs[1:]
        assert len(refs) >= 2
    finally:
        os.remove(best_file)


def test_eval_refs_no_best_file():
    """No best_step.txt: auto-creates from oldest ckpt."""
    import os
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


def test_eval_refs_spread_across_history():
    """Larger ref_count covers more of the training history."""
    ckpts = list(range(10, 101, 10))
    refs = select_eval_references(ckpts, 105, ".", ref_count=5)
    # best=10, targets: 105//6≈17→20, 105*2//6=35→30 or 40, 105*3//6=52→50, 105*4//6=70→70
    assert refs[0] == 10
    assert len(refs) >= 4


def test_equal_spacing_ref_count_4():
    """ref_count=4 → 3 milestones, boundaries at step/5 fractions."""
    ckpts = list(range(10, 91, 10))  # [10..90]
    # step=95, ref_count=4, n_segments=3, step//(n_segments+1)=95//4=23
    # targets: 23, 47, 71 → nearest: 20, 50, 70
    refs = select_eval_references(ckpts, 95, ".", ref_count=4)
    assert refs[0] == 10   # best
    assert refs[1] == 20   # nearest to 23
    assert refs[2] == 50   # nearest to 47 (40=7, 50=3)
    assert refs[3] == 70   # nearest to 71 (70=1, 80=9)
    assert len(refs) == 4


def test_equal_spacing_early_training():
    """Early training with few checkpoints: still evenly distributed."""
    ckpts = [5, 10]
    refs = select_eval_references(ckpts, 15, ".", ref_count=2)
    # best=5, 1 milestone: target=15//3=5, nearest=5 but 5 is best → fallback=10
    # Actually best=5 (oldest), then target=15//3=5 → 5 is best (in refs) →
    # no exact match, nearest unchecked is 10 (|10-5|=5). refs=[5, 10]
    assert 5 in refs and 10 in refs
    assert len(refs) == 2


def test_equal_spacing_ref_count_1():
    """ref_count=1 → only best, no milestones."""
    ckpts = list(range(10, 51, 10))
    refs = select_eval_references(ckpts, 55, ".", ref_count=1)
    assert len(refs) == 1
    assert 10 in refs  # best = oldest

def test_eval_refs_count_never_exceeds():
    """refs count never exceeds ref_count."""
    import os
    ckpts = list(range(10, 201, 10))
    for n in [1, 2, 3, 5, 10]:
        best_file = os.path.join(".", "best_step.txt")
        if os.path.exists(best_file):
            os.remove(best_file)
        refs = select_eval_references(ckpts, 205, ".", ref_count=n)
        assert len(refs) <= n
    if os.path.exists(best_file):
        os.remove(best_file)


def test_stable_qdr():
    """_stable_qdr skips children with ≤1 visit (forced-explore noise)."""
    root = Node(None, 0, 1.0)
    root.explore_count = 10
    root.total_reward = 1.0
    root.draw_reward = 0.3

    # Child with >1 visit — should be included
    c1 = Node(1, 0, 0.5)
    c1.explore_count = 5
    c1.total_reward = 3.0
    c1.draw_reward = 0.2
    root.children.append(c1)

    # Child with 1 visit — should be excluded (forced-explore noise)
    c2 = Node(2, 0, 0.3)
    c2.explore_count = 1
    c2.total_reward = -999.0  # noise that would corrupt Q
    c2.draw_reward = 0.0
    root.children.append(c2)

    # Child with 0 visits — should be excluded
    c3 = Node(3, 0, 0.2)
    c3.explore_count = 0
    c3.total_reward = 0.0
    root.children.append(c3)

    q, dr = _stable_qdr(root)
    # Only c1 contributes: q = 3.0/5 = 0.6, dr = 0.2/5 = 0.04
    assert abs(q - 0.6) < 1e-9, f"expected q=0.6, got {q}"
    assert abs(dr - 0.04) < 1e-9, f"expected dr=0.04, got {dr}"


def test_stable_qdr_fallback():
    """_stable_qdr falls back to root values when no child qualifies."""
    root = Node(None, 0, 1.0)
    root.explore_count = 5
    root.total_reward = 2.0
    root.draw_reward = 0.5

    # max_n=0 → immediate fallback before threshold check
    q, dr = _stable_qdr(root)
    assert q == root.q_value and dr == root.draw_rate, \
        f"expected fallback to root Q/DR, got q={q} dr={dr}"


def test_stable_qdr_no_children():
    """_stable_qdr with empty children falls back to root."""
    root = Node(None, 0, 1.0)
    root.explore_count = 3
    root.total_reward = 1.5
    root.draw_reward = 0.1

    q, dr = _stable_qdr(root)
    assert q == root.q_value and dr == root.draw_rate


def test_assign_players_main_only():
    """Without best/opp models, always main vs main."""
    rng = np.random.RandomState(42)
    main = object(); best = object(); opp = None
    seen = set()
    for _ in range(20):
        p0, p1, ub, uo = assign_players(rng, main, best, opp,
                                         best_model_prob=0, random_opponent_prob=0)
        assert p0 is main and p1 is main
        assert not ub and not uo
        seen.add((p0, p1))
    assert len(seen) == 1  # always the same


def test_assign_players_best_prob_1():
    """best_model_prob=1 → always main vs best."""
    rng = np.random.RandomState(42)
    main = object(); best = object(); opp = None
    for _ in range(20):
        p0, p1, ub, uo = assign_players(rng, main, best, opp,
                                         best_model_prob=1.0, random_opponent_prob=0)
        assert ub and not uo
        assert (p0 is main and p1 is best) or (p0 is best and p1 is main)


def test_assign_players_opp_prob_1():
    """random_opponent_prob=1, best_prob=0 → always main vs opp."""
    rng = np.random.RandomState(42)
    main = object(); best = object(); opp = object()
    for _ in range(20):
        p0, p1, ub, uo = assign_players(rng, main, best, opp,
                                         best_model_prob=0, random_opponent_prob=1.0)
        assert not ub and uo
        assert (p0 is main and p1 is opp) or (p0 is opp and p1 is main)


def test_assign_players_colour_swap():
    """When p1≠main, colours are sometimes swapped."""
    rng = np.random.RandomState(42)
    main = object(); best = object(); opp = None
    black_best = 0  # p0 is main, p1 is best
    white_best = 0  # p0 is best, p1 is main
    for _ in range(200):
        p0, p1, _, _ = assign_players(rng, main, best, opp,
                                       best_model_prob=1.0, random_opponent_prob=0)
        if p0 is main and p1 is best:
            black_best += 1
        else:
            white_best += 1
    # ~50% each, with some tolerance
    assert 70 < black_best < 130 and 70 < white_best < 130, \
        f"swap bias: black={black_best} white={white_best}"


def test_assign_players_explicit_use():
    """Explicit use_best/use_opp overrides random draw."""
    rng = np.random.RandomState(42)
    main = object(); best = object(); opp = object()
    p0, p1, ub, uo = assign_players(rng, main, best, opp,
                                     use_best=True, use_opp=False)
    assert ub and not uo
    assert (p0 is main and p1 is best) or (p0 is best and p1 is main)

    p0, p1, ub, uo = assign_players(rng, main, best, opp,
                                     use_best=False, use_opp=True)
    assert not ub and uo
    assert (p0 is main and p1 is opp) or (p0 is opp and p1 is main)


def main():
    tests = [
        test_find_latest_empty, test_find_latest_single,
        test_find_latest_multiple, test_find_latest_ignores_other_files,
        test_load_base_config_defaults, test_load_base_config_from_json,
        test_load_base_config_missing_file,
        test_wstats_accum, test_wstats_reset, test_wstats_empty,
        test_backward_compat_imports,
        test_stable_qdr,
        test_stable_qdr_fallback,
        test_stable_qdr_no_children,
        test_assign_players_main_only,
        test_assign_players_best_prob_1,
        test_assign_players_opp_prob_1,
        test_assign_players_colour_swap,
        test_assign_players_explicit_use,
        test_eval_refs_empty, test_eval_refs_one_ckpt,
        test_eval_refs_evenly_spaced, test_eval_refs_best_not_duplicated,
        test_eval_refs_no_best_file, test_eval_refs_spread_across_history,
        test_equal_spacing_ref_count_4, test_equal_spacing_early_training,
        test_equal_spacing_ref_count_1,
        test_eval_refs_count_never_exceeds,
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
