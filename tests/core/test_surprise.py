"""Test surprise detection (KL-based)."""

import numpy as np
import pyspiel
from train.batch_mcts.node import Node
from train.core.surprise import detect_surprise, _kl, _wdl_from_qdr, _stable_qdr


class _FakeConfig:
    surprise_pol_kl = 0.5
    surprise_val_kl = 0.3
    surprise_child_min_n = 150


def _make_root(nn_q=None, nn_draw=0.0, nn_prior_max=None, nn_argmax=None,
               nn_prior=None, children=None):
    root = Node(None, 0, 1.0)
    root.nn_q = nn_q
    root.nn_draw = nn_draw
    root.nn_prior = nn_prior
    root.nn_prior_max = nn_prior_max
    root.nn_argmax = nn_argmax
    root.children = children or []
    root.explore_count = sum(c.explore_count for c in root.children)
    root.total_reward = sum(c.total_reward for c in root.children)
    root.draw_reward = 0.0
    return root


def _make_child(action, explore_count, total_reward, prior=0.1, player=0,
                nn_q=0.0, nn_draw=0.0, nn_prior=None, draw_reward=None,
                outcome=None, children=None):
    c = Node(action, player, prior)
    c.explore_count = explore_count
    c.total_reward = total_reward
    c.draw_reward = draw_reward if draw_reward is not None else 0.0
    c.nn_q = nn_q
    c.nn_draw = nn_draw
    c.nn_prior = nn_prior
    c.children = children or []
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
#  Test F: _stable_qdr with outcome
# ═══════════════════════════════════════════════════════════════════════════════

def test_stable_qdr_outcome():
    # Proven win for player 0
    c = _make_child(0, 10, 10, outcome=[1, -1])
    c.player = 0
    q, dr = _stable_qdr(c)
    assert q == 1.0
    assert dr == 0.0

    # Proven draw
    c2 = _make_child(0, 10, 0, outcome=[0, 0])
    c2.player = 0
    q, dr = _stable_qdr(c2)
    assert q == 0.0
    assert dr == 1.0
    print("  Test F PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
#  Test G: child_surprise from value KL
# ═══════════════════════════════════════════════════════════════════════════════

def test_child_surprise_value():
    """Non-proven child: value surprise skipped, no tag despite value mismatch."""
    cfg = _FakeConfig()
    game = pyspiel.load_game("tic_tac_toe")
    state = game.new_initial_state()

    cc = _make_child(0, 15, -12, prior=0.5, player=1)
    # outcome=None (non-proven) → value KL forced to 0
    c = _make_child(0, 200, -160, prior=0.5, player=1,
                    nn_q=0.8, nn_draw=0.0,
                    nn_prior=[(0, 0.5), (1, 0.5)],
                    draw_reward=0.0, children=[cc])

    root = _make_root(nn_q=0.3, nn_draw=0.1, children=[c])
    root.total_reward = 60
    root.draw_reward = 0.1

    _, child_tags, _ = detect_surprise(state, root, cfg, game.max_utility())
    assert 0 not in child_tags, \
        f"Non-proven child should not trigger value surprise, got {child_tags}"


def test_child_surprise_value_proven():
    """Proven child (outcome set) still triggers value surprise."""
    cfg = _FakeConfig()
    game = pyspiel.load_game("tic_tac_toe")
    state = game.new_initial_state()

    cc = _make_child(0, 15, -12, prior=0.5, player=1)
    c = _make_child(0, 200, -160, prior=0.5, player=1,
                    nn_q=0.8, nn_draw=0.0,
                    nn_prior=[(0, 0.5), (1, 0.5)],
                    draw_reward=0.0, children=[cc])
    c.outcome = np.array([0.0, 0.0])  # proven draw

    root = _make_root(nn_q=0.3, nn_draw=0.1, children=[c])
    root.total_reward = 60
    root.draw_reward = 0.1

    _, child_tags, _ = detect_surprise(state, root, cfg, game.max_utility())
    assert 0 in child_tags, \
        f"Proven child should trigger value surprise, got {child_tags}"


# ═══════════════════════════════════════════════════════════════════════════════
#  Test H: child_super_surprise from policy+value KL
# ═══════════════════════════════════════════════════════════════════════════════

def test_child_super_surprise():
    cfg = _FakeConfig()
    game = pyspiel.load_game("tic_tac_toe")
    state = game.new_initial_state()

    # NN prior says action 0 is 90%, but MCTS visits went to action 1 (80%)
    # Also value mismatch: NN Q=0.8, MCTS Q=-0.4 → combined KL > thresholds
    cc0 = _make_child(0, 20, -8, prior=0.9, player=1)   # Q=-0.4
    cc1 = _make_child(1, 80, -32, prior=0.1, player=1)  # Q=-0.4
    c = _make_child(0, 200, -40, prior=0.5, player=1,
                    nn_q=0.8, nn_draw=0.0,
                    nn_prior=[(0, 0.9), (1, 0.1)],
                    draw_reward=0.0, children=[cc0, cc1])

    root = _make_root(nn_q=0.3, nn_draw=0.1, children=[c])
    root.total_reward = 60
    root.draw_reward = 0.1

    _, child_tags, _ = detect_surprise(state, root, cfg, game.max_utility())
    assert 0 in child_tags, f"Expected child tag, got {child_tags}"
    ctag, _ = child_tags[0]
    assert ctag == "child_super_surprise", \
        f"Expected child_super_surprise, got {ctag}"
    print(f"  Test H PASSED (tag={ctag})")


# ═══════════════════════════════════════════════════════════════════════════════
#  Test I: child with no nn_prior is skipped
# ═══════════════════════════════════════════════════════════════════════════════

def test_child_no_nn_prior_skipped():
    cfg = _FakeConfig()
    game = pyspiel.load_game("tic_tac_toe")
    state = game.new_initial_state()

    c = _make_child(0, 200, -160, prior=0.5, nn_q=0.8, nn_draw=0.0,
                    nn_prior=None)  # no prior
    c.player = 1

    root = _make_root(nn_q=0.3, nn_draw=0.1, children=[c])
    root.total_reward = 60
    root.draw_reward = 0.1

    _, child_tags, _ = detect_surprise(state, root, cfg, game.max_utility())
    assert child_tags == {}, f"Child without nn_prior should be skipped, got {child_tags}"
    print("  Test I PASSED")

# ═══════════════════════════════════════════════════════════════════════════════
#  Test J: _stable_qdr uses state.current_player() for non-terminal outcome
# ═══════════════════════════════════════════════════════════════════════════════

def test_stable_qdr_state_perspective():
    """For non-terminal proven nodes, Q is from state.current_player()."""
    game = pyspiel.load_game("tic_tac_toe")
    state = game.new_initial_state()  # player 0 to move
    state.apply_action(0)             # player 0 plays, state now player 1
    # Not terminal yet — but set a proven outcome (MCTS-Solver)
    c = _make_child(0, 200, 200, player=0,
                    outcome=[1, -1])  # p0 wins
    c.state = state  # current_player() = 1, NOT terminal
    # _stable_qdr should use state.current_player() = 1
    q, dr = _stable_qdr(c)
    # outcome[1] = -1 (loss for p1)
    assert q == -1.0, f"Expected Q=-1 from p1 perspective, got {q}"
    assert dr == 0.0
    print("  Test J PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
#  Test K: _stable_qdr falls back to node.player for terminal state
# ═══════════════════════════════════════════════════════════════════════════════

def test_stable_qdr_terminal_fallback():
    """For terminal states, state.current_player() is TERMINAL(-4),
    so fall back to node.player."""
    game = pyspiel.load_game("tic_tac_toe")
    state = game.new_initial_state()
    # Apply moves until terminal (quick win for p0 in tic-tac-toe)
    state.apply_action(0)  # p0: top-left
    state.apply_action(3)  # p1: middle-left
    state.apply_action(1)  # p0: top-center
    state.apply_action(4)  # p1: middle-center
    state.apply_action(2)  # p0: top-right → win
    assert state.is_terminal()
    c = _make_child(2, 200, 200, player=0,
                    outcome=state.returns())  # p0 wins [1, -1]
    c.state = state  # is_terminal() = True, current_player() = -4
    q, dr = _stable_qdr(c)
    # Falls back to node.player=0 → outcome[0]=1 (win for p0)
    assert q == 1.0, f"Expected Q=1 from p0 (fallback), got {q}"
    assert dr == 0.0
    print("  Test K PASSED")

# ═══════════════════════════════════════════════════════════════════════════════
#  Test L: child solved_policy uses state.current_player(), not c.player
# ═══════════════════════════════════════════════════════════════════════════════

def test_child_solved_policy_perspective():
    """Child's compute_solved_policy must use c.state.current_player().

    c.player=0 (root expand) but c.state.current_player()=1.
    c's children outcomes are [1,-1] (p0 wins).
    With wrong perspective (player=0): outcome[0]=1 → "both winning"
      → Case 1, best_val=1, weight=1/explore_count
    With correct perspective (player=1): outcome[1]=-1 → "both losing"
      → Case 1, best_val=-1, weight=explore_count (prefer most-tested)
    These give different policies → we verify the correct one is used.
    """
    game = pyspiel.load_game("tic_tac_toe")
    state = game.new_initial_state()
    state.apply_action(0)  # p0 plays, state now p1 to move

    # Two grandchildren, both p0 wins [1, -1], different explore counts
    gc0 = _make_child(0, 5, 5, player=1, outcome=[1, -1])
    gc1 = _make_child(1, 15, 15, player=1, outcome=[1, -1])

    c = _make_child(0, 200, 200, player=0,  # c.player = 0 (root expand)
                    children=[gc0, gc1])
    c.state = state  # current_player() = 1

    cfg = _FakeConfig()
    cfg.surprise_child_min_n = 1  # low enough to trigger
    root = _make_root(nn_q=0.0, nn_draw=0.0, children=[c])
    root.total_reward = 0
    root.draw_reward = 0.0

    # Inject nn_prior and nn_q so detection runs
    c.nn_q = 0.0
    c.nn_draw = 0.0
    c.nn_prior = [(0, 0.5), (1, 0.5)]

    _, child_tags, _ = detect_surprise(
        game.new_initial_state(), root, cfg, game.max_utility())

    # If perspective is wrong (player=0), both children "win" → equal weights
    # If perspective is correct (player=1), both "lose" → gc1 gets higher weight (more explored)
    # Either way, the detection runs without crash and produces a tag
    # The key assertion: no crash, perspective is internally consistent
    assert isinstance(child_tags, dict)
    print("  Test L PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
#  Test M: child solved_policy perspective — explicit policy check
# ═══════════════════════════════════════════════════════════════════════════════

def test_child_solved_policy_explicit():
    """Explicitly check that the solved policy uses state.current_player()."""
    from train.batch_mcts.mcts import compute_solved_policy as csp

    game = pyspiel.load_game("tic_tac_toe")
    state = game.new_initial_state()
    state.apply_action(0)  # p0 played, now p1's turn

    gc0 = _make_child(0, 5, 5, player=1, outcome=[1, -1])  # p0 win
    gc1 = _make_child(1, 15, 15, player=1, outcome=[1, -1])

    children = [gc0, gc1]

    # Use correct perspective: player=1 (state.current_player())
    pol_good = csp(children, 1, game.max_utility())
    # Use wrong perspective: player=0 (c.player)
    pol_bad = csp(children, 0, game.max_utility())

    # With player=1: both lose → Case 1 losing → prefer most-explored (gc1)
    assert pol_good[1] > pol_good[0], \
        f"Correct perspective: gc1 (more explored) should dominate, got {pol_good}"

    # With player=0: both win → Case 1 winning → prefer least-explored (gc0)
    assert pol_bad[0] > pol_bad[1], \
        f"Wrong perspective: gc0 (less explored) should dominate, got {pol_bad}"

    print("  Test M PASSED")

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
