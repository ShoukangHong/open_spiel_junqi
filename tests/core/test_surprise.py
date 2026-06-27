"""Test surprise detection (KL-based)."""

import numpy as np
import pyspiel
from train.batch_mcts.node import Node
from train.core.surprise import detect_surprise, _kl, _wdl_from_qdr


class _FakeConfig:
    surprise_pol_kl = 0.5
    surprise_val_kl = 0.3
    surprise_child_min_n = 150


def _make_root(nn_q=None, nn_draw=0.0, nn_prior_max=None, nn_argmax=None,
               children=None):
    root = Node(None, 0, 1.0)
    root.nn_q = nn_q
    root.nn_draw = nn_draw
    root.nn_prior_max = nn_prior_max
    root.nn_argmax = nn_argmax
    root.children = children or []
    root.explore_count = sum(c.explore_count for c in root.children)
    root.total_reward = sum(c.total_reward for c in root.children)
    root.draw_reward = 0.0
    return root


def _make_child(action, explore_count, total_reward, prior=0.1,
                nn_q=0.0, nn_draw=0.0, outcome=None):
    c = Node(action, 0, prior)
    c.explore_count = explore_count
    c.total_reward = total_reward
    c.draw_reward = 0.0
    c.nn_q = nn_q
    c.nn_draw = nn_draw
    if outcome is not None:
        c.outcome = np.array(outcome, dtype=np.float64)
    return c


# ═══════════════════════════════════════════════════════════════════════════════
#  Test A: no NN cache → no surprise
# ═══════════════════════════════════════════════════════════════════════════════

def test_no_cache():
    cfg = _FakeConfig()
    game = pyspiel.load_game("tic_tac_toe")
    state = game.new_initial_state()
    root = _make_root(nn_q=None)
    tag, child_tags, _ = detect_surprise(state, root, cfg, game.max_utility())
    assert tag == ""
    assert child_tags == {}
    print("  Test A PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
#  Test B: no surprise when MCTS and NN agree
# ═══════════════════════════════════════════════════════════════════════════════

def test_no_surprise():
    cfg = _FakeConfig()
    game = pyspiel.load_game("tic_tac_toe")
    state = game.new_initial_state()
    # MCTS and NN both agree: Q=0.3, same policy distribution
    c0 = _make_child(0, 500, 150, prior=0.6, nn_q=0.3)
    c1 = _make_child(1, 500, 150, prior=0.4, nn_q=0.3)
    root = _make_root(nn_q=0.3, nn_draw=0.1, children=[c0, c1])
    root.total_reward = 300
    root.draw_reward = 0.1
    tag, _, _ = detect_surprise(state, root, cfg, game.max_utility())
    assert tag == "", f"Expected no surprise, got {tag}"
    print("  Test B PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
#  Test C: value KL surprise → "surprise"
# ═══════════════════════════════════════════════════════════════════════════════

def test_value_surprise():
    cfg = _FakeConfig()
    game = pyspiel.load_game("tic_tac_toe")
    state = game.new_initial_state()
    # NN says WDL≈[0.9,0,0.1] (Q=0.8,d=0), MCTS found WDL≈[0.1,0,0.9] (Q=-0.8)
    # KL will be large
    c0 = _make_child(0, 500, -400, prior=0.6, nn_q=0.8)
    c1 = _make_child(1, 500, -400, prior=0.4, nn_q=0.8)
    root = _make_root(nn_q=0.8, nn_draw=0.0, children=[c0, c1])
    root.total_reward = -400  # Q ≈ -0.4, wait this gives KL with nn_q=0.8
    root.draw_reward = 0.0
    # Compute actual KL to verify
    nn = _wdl_from_qdr(0.8, 0.0)
    mcts = _wdl_from_qdr(-0.4, 0.0)
    kl = _kl(nn, mcts)
    print(f"  Value KL: NN{nn} vs MCTS{mcts} → KL={kl:.4f}")
    tag, _, _ = detect_surprise(state, root, cfg, game.max_utility())
    assert tag, f"Expected surprise, got {tag}"
    print("  Test C PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
#  Test D: policy KL surprise → "strong_surprise"
# ═══════════════════════════════════════════════════════════════════════════════

def test_policy_surprise():
    cfg = _FakeConfig()
    game = pyspiel.load_game("tic_tac_toe")
    state = game.new_initial_state()
    # NN prior heavily favors action 0 (0.9/0.1), MCTS found action 1 better
    # (solved policy dominated by action 1 due to high visits)
    c0 = _make_child(0, 100, 30, prior=0.9, nn_q=0.3)
    c1 = _make_child(1, 900, 270, prior=0.1, nn_q=0.3)
    root = _make_root(nn_q=0.3, nn_draw=0.0, children=[c0, c1])
    root.total_reward = 300
    root.draw_reward = 0.0
    tag, _, _ = detect_surprise(state, root, cfg, game.max_utility())
    assert tag == "super_surprise", f"Expected super_surprise, got {tag}"
    print("  Test D PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
#  Test E: WDL helper functions
# ═══════════════════════════════════════════════════════════════════════════════

def test_wdl_helpers():
    # Q=0.8, draw=0 → w≈0.9, l≈0.1
    wdl = _wdl_from_qdr(0.8, 0.0)
    assert abs(wdl[0] - 0.9) < 0.01
    assert abs(wdl[2] - 0.1) < 0.01

    # Q=0, draw=1.0 → pure draw
    wdl = _wdl_from_qdr(0.0, 1.0)
    assert abs(wdl[1] - 1.0) < 0.01

    # Symmetric KL between identical distributions → 0
    p = np.array([0.5, 0.3, 0.2])
    assert _kl(p, p) < 1e-9
    print("  Test E PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    tests = [
        ("A", "no NN cache", test_no_cache),
        ("B", "no surprise", test_no_surprise),
        ("C", "value KL surprise", test_value_surprise),
        ("D", "policy KL strong", test_policy_surprise),
        ("E", "WDL helpers", test_wdl_helpers),
    ]
    failed = 0
    for label, desc, fn in tests:
        try:
            fn()
        except Exception as e:
            print(f"\n  Test {label} ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    print(f"\n{'=' * 50}")
    print(f"  {'ALL PASSED' if failed == 0 else f'{failed} FAILED'}")
    print(f"{'=' * 50}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
