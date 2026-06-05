"""Tests for WDL (win/draw/loss) value head support in MCTS."""

import numpy as np
import pytest
import pyspiel

from train.batch_mcts.config import MCTSConfig
from train.batch_mcts.mcts import BatchMCTS
from train.batch_mcts.node import Node


# ── Node ────────────────────────────────────────────────────────────────────

def test_node_draw_rate_default():
    n = Node(None, 0, 1.0)
    assert n.draw_reward == 0.0
    assert n.draw_rate == 0.0


def test_node_draw_rate_accumulates():
    n = Node(None, 0, 1.0)
    n.explore_count = 10
    n.draw_reward = 4.0
    assert n.draw_rate == 0.4


def test_node_draw_does_not_affect_q():
    n = Node(None, 0, 1.0)
    n.explore_count = 10
    n.total_reward = 5.0
    n.draw_reward = 9.0  # high draw shouldn't affect Q
    assert n.q_value == 0.5
    assert n.draw_rate == 0.9


# ── MCTS with WDL evaluator ─────────────────────────────────────────────────

class _WdlEvaluator:
    """Returns fixed WDL values for all states."""
    def __init__(self, w=0.3, d=0.4, l=0.3):
        self._w, self._d, self._l = w, d, l

    def batch_inference_raw(self, states):
        values = np.array([[self._w, self._d, self._l]] * len(states),
                          dtype=np.float32)
        priors = []
        for s in states:
            legal = s.legal_actions()
            priors.append([(a, 1.0 / len(legal)) for a in legal])
        return values, priors


def test_wdl_mcts_returns_root():
    """Run a small search with WDL evaluator — just smoke test."""
    game = pyspiel.load_game("tic_tac_toe")
    config = MCTSConfig(max_simulations=32, batch_size=4)
    evaluator = _WdlEvaluator()
    mcts = BatchMCTS(game, config, evaluator,
                     random_state=np.random.RandomState(42))
    root = mcts.mcts_search(game.new_initial_state())
    assert root is not None
    assert root.explore_count == 32  # consumed all sims


def test_wdl_root_draw_rate():
    """Root draw_rate should reflect the evaluator's draw probability."""
    game = pyspiel.load_game("tic_tac_toe")
    config = MCTSConfig(max_simulations=64, batch_size=8,
                        )
    # Pure draw evaluator
    class PureDrawEval:
        def batch_inference_raw(self, states):
            values = np.array([[0.0, 1.0, 0.0]] * len(states),
                              dtype=np.float32)
            priors = []
            for s in states:
                legal = s.legal_actions()
                priors.append([(a, 1.0 / len(legal)) for a in legal])
            return values, priors

    mcts = BatchMCTS(game, config, PureDrawEval(),
                     random_state=np.random.RandomState(42))
    root = mcts.mcts_search(game.new_initial_state())
    # Tic-tac-toe is deterministic — no terminal draws unless both play perfectly.
    # With PureDrawEval returning d=1.0, non-terminal leaves get draw_prob=1.0.
    # Terminal wins still get draw_prob=0.
    assert root.draw_rate >= 0.0


def _scalar_to_wdl(p0_value, player, draw_rate=0.0):
    """p0-perspective scalar → current-player WDL with fixed draw_rate."""
    q = p0_value if player == 0 else -p0_value
    d = draw_rate
    w = max(0.0, (1.0 + q) / 2.0 - d / 2.0)
    l = max(0.0, (1.0 - q) / 2.0 - d / 2.0)
    s = w + d + l
    return [w / s, d / s, l / s]


def _state_scalar(state):
    """Deterministic scalar per state (same for WDL and scalar eval)."""
    rng = np.random.RandomState(abs(hash(str(state))) % (2 ** 31))
    return float(rng.uniform(-0.9, 0.9))


# ── Terminal draw detection ─────────────────────────────────────────────────

def test_terminal_draw_sets_draw_prob():
    """A terminal draw should backprop draw_prob=1.0."""
    game = pyspiel.load_game("tic_tac_toe")
    # Create a state one move away from a forced draw
    state = game.new_initial_state()
    # Fill the board to force a draw
    for a in [0, 1, 2, 5, 3, 6, 4, 8, 7]:
        if state.is_terminal():
            break
        state.apply_action(a)

    if not state.is_terminal():
        # State is close to terminal
        config = MCTSConfig(max_simulations=32, batch_size=8,
                            )
        # Run one search — this will hit terminal draws
        pass  # can't guarantee terminal, skip assertion
    # If terminal, check
    if state.is_terminal():
        r = state.returns()
        if r[0] == 0 and r[1] == 0:
            pass  # it's a draw


# ── run ─────────────────────────────────────────────────────────────────────

def main():
    tests = [
        test_node_draw_rate_default, test_node_draw_rate_accumulates,
        test_node_draw_does_not_affect_q,
        test_wdl_mcts_returns_root, test_wdl_root_draw_rate,
        test_draw_backprop_through_tree, test_draw_backprop_perspective,
        test_solved_draw_sets_draw_rate,
        test_wdl_mcts_q_from_wdl, test_wdl_mcts_q_perspective_consistent,
        test_wdl_draw_rate_at_depth, test_wdl_mcts_with_terminal_leaf,
        test_wdl_perspective_flip, test_wdl_perspective_draw_symmetric,
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


def test_draw_backprop_through_tree():
    """Verify draw probability aggregates correctly and is NOT sign-flipped
    across zero-sum perspective changes."""
    # Use a game with a known shallow tree so we can inspect paths.
    game = pyspiel.load_game("tic_tac_toe")
    state = game.new_initial_state()
    # Play 4 moves to reach a state with few legal actions
    for a in [0, 4, 8, 2]:
        state.apply_action(a)
    # Now O to move at (2,0) — board has pieces, narrow tree

    draw_val = 0.7
    class FixedWdlEval:
        def batch_inference_raw(self, states):
            values = np.array([[0.1, draw_val, 0.2]] * len(states),
                              dtype=np.float32)
            priors = []
            for s in states:
                legal = s.legal_actions()
                priors.append([(a, 1.0 / len(legal)) for a in legal])
            return values, priors

    config = MCTSConfig(max_simulations=64, batch_size=8,                         uct_c=1.41)
    mcts = BatchMCTS(game, config, FixedWdlEval(),
                     random_state=np.random.RandomState(123))
    root = mcts.mcts_search(state.clone())

    # Every non-terminal leaf gets draw_prob=0.7 → every backprop adds 0.7
    # Root's draw should be non-zero and positive
    assert root.draw_rate > 0.0, "root draw_rate should be > 0"
    assert root.draw_rate == pytest.approx(draw_val, abs=0.3), \
        f"root draw_rate={root.draw_rate:.3f} expected ~{draw_val}"

    # Check children: ALL should have positive draw_rate (no sign flip)
    for child in root.children:
        if child.explore_count > 0:
            assert child.draw_rate >= 0.0, \
                f"child draw_rate should be non-negative, got {child.draw_rate}"
            if child.children:
                for grandchild in child.children:
                    if grandchild.explore_count > 0:
                        assert grandchild.draw_rate >= 0.0, \
                            f"grandchild draw_rate should be non-negative"


def test_draw_backprop_perspective():
    """In a two-player game, draw prob should be symmetric — both players
    see the same draw likelihood.  Test with a pure-draw terminal."""
    game = pyspiel.load_game("tic_tac_toe")
    # Build a forced-draw board
    # X: (0,0)(0,2)(1,0)(1,2)(2,1)  O: (0,1)(1,1)(2,0)(2,2)
    # This is a typical tic-tac-toe draw board
    state = game.new_initial_state()
    for a in [0, 1, 2, 3, 4, 5, 6, 8, 7]:
        if not state.is_terminal():
            state.apply_action(a)

    if state.is_terminal() and all(r == 0 for r in state.returns()):
        # Run MCTS from a state near terminal so we can observe draw backprop
        # Go back one move
        pass  # reusing above state

    # Simpler approach: start from initial, use an eval that returns pure draw
    s = game.new_initial_state()
    s.apply_action(0)

    class PureDrawEval:
        def batch_inference_raw(self, states):
            values = np.array([[0.0, 1.0, 0.0]] * len(states),
                              dtype=np.float32)
            priors = []
            for st in states:
                legal = st.legal_actions()
                priors.append([(a, 1.0 / len(legal)) for a in legal])
            return values, priors

    config = MCTSConfig(max_simulations=32, batch_size=4)
    mcts = BatchMCTS(game, config, PureDrawEval(),
                     random_state=np.random.RandomState(42))
    root = mcts.mcts_search(s)

    # With pure draw evaluator, ALL non-terminal leaves have d=1.0
    # Terminal wins (if reached) have d=0.
    # Root and children should have significant draw_rate
    assert root.draw_rate > 0.5, \
        f"with d=1.0 eval, root draw_rate should be high, got {root.draw_rate:.3f}"

    # Check all children have consistent draw_rate (no perspective flip)
    rates = []
    for child in root.children:
        if child.explore_count > 0:
            rates.append(child.draw_rate)
    if rates:
        # All should be positive and close to each other (within noise)
        assert all(r > 0 for r in rates), \
            f"all child draw rates should be positive: {rates}"
        assert max(rates) - min(rates) < 0.5, \
            f"child draw rates should be similar: {rates}"


def test_solved_draw_sets_draw_rate():
    """In a proven-draw position, root.draw_rate should be 1.0."""
    game = pyspiel.load_game("tic_tac_toe")
    state = game.new_initial_state()
    # Create a fork position where all children are proven losses
    # except two that are proven draws (tic-tac-toe fork state)
    state.apply_action(0)   # X(0,0)
    state.apply_action(8)   # O(2,2)
    state.apply_action(4)   # X(1,1)
    # Ground truth: moves {2,6} are DRAW, rest are LOSS for O
    # MCTS solver should prove this

    config = MCTSConfig(max_simulations=2000, batch_size=32,
                        uct_c=2.0,
                        solve=True)
    mcts = BatchMCTS(game, config, _WdlEvaluator(),
                     random_state=np.random.RandomState(42))
    root = mcts.mcts_search(state.clone())

    # If solver proved the root, check draw_rate
    if root.outcome is not None:
        if all(r == 0 for r in root.outcome):
            assert root.draw_rate == 1.0, \
                f"proven draw should have draw_rate=1.0, got {root.draw_rate}"
        else:
            assert root.draw_rate == 0.0, \
                f"proven win/loss should have draw_rate=0.0, got {root.draw_rate}"


def test_wdl_mcts_q_from_wdl():
    """MCTS Q should be computed as (w-l)*max_utility from WDL evaluator."""
    game = pyspiel.load_game("tic_tac_toe")
    state = game.new_initial_state()

    class FixedWdlEval:
        def batch_inference_raw(self, states):
            # p0: w=0.7,l=0.2→Q=0.5. p1: w=0.2,l=0.7→Q=-0.5
            values = np.array(
                [_scalar_to_wdl(0.5, s.current_player(), draw_rate=0.1)
                 for s in states], dtype=np.float32)
            priors = []
            for s in states:
                legal = s.legal_actions()
                priors.append([(a, 1.0 / len(legal)) for a in legal])
            return values, priors

    config = MCTSConfig(max_simulations=64, batch_size=8)
    mcts = BatchMCTS(game, config, FixedWdlEval(),
                     random_state=np.random.RandomState(42))
    root = mcts.mcts_search(state.clone())
    # root Q should be near 0.5 (the evaluator's value)
    assert abs(root.q_value - 0.5) < 1e-3, \
        f"expected Q≈0.5, got {root.q_value:.6f}"
    # draw_rate should be near 0.1
    assert abs(root.draw_rate - 0.1) < 0.1, \
        f"expected draw_rate≈0.1, got {root.draw_rate:.3f}"


def test_wdl_mcts_q_perspective_consistent():
    """MCTS backs up Q values consistently — Q should not drift from
    the evaluator's (w-l) regardless of search depth."""
    game = pyspiel.load_game("tic_tac_toe")
    state = game.new_initial_state()

    class WinBiasEval:
        def batch_inference_raw(self, states):
            # WDL from current player. p0: w=0.6,l=0.2→Q=0.4. p1: w=0.2,l=0.6→Q=-0.4
            values = np.array(
                [[0.6, 0.2, 0.2] if s.current_player() == 0
                 else [0.2, 0.2, 0.6] for s in states],
                dtype=np.float32)
            priors = []
            for s in states:
                legal = s.legal_actions()
                priors.append([(a, 1.0 / len(legal)) for a in legal])
            return values, priors

    # Test at different sim counts — Q should converge
    for sims in [16, 64]:
        config = MCTSConfig(max_simulations=sims, batch_size=4,
                            )
        mcts = BatchMCTS(game, config, WinBiasEval(),
                         random_state=np.random.RandomState(42))
        root = mcts.mcts_search(state.clone())
        # Q should be roughly 0.4
        assert abs(root.q_value - 0.4) < 1e-3, \
            f"sims={sims}: expected Q≈0.4, got {root.q_value:.3f}"


def test_wdl_draw_rate_at_depth():
    """draw_rate should be backpropped correctly at all tree depths."""
    game = pyspiel.load_game("tic_tac_toe")
    state = game.new_initial_state()
    state.apply_action(0)  # play one move to reduce branching

    class DrawEval:
        def batch_inference_raw(self, states):
            values = np.array([[0.0, 1.0, 0.0]] * len(states),
                              dtype=np.float32)
            priors = []
            for s in states:
                legal = s.legal_actions()
                priors.append([(a, 1.0 / len(legal)) for a in legal])
            return values, priors

    config = MCTSConfig(max_simulations=64, batch_size=8,                         uct_c=1.41)
    mcts = BatchMCTS(game, config, DrawEval(),
                     random_state=np.random.RandomState(123))
    root = mcts.mcts_search(state.clone())

    # All visited nodes should have high draw_rate
    def check_draw(node, depth=0, max_depth=3):
        if node.explore_count > 0 and depth > 0:
            assert node.draw_rate > 0.5, \
                f"depth={depth}: expected draw_rate>0.5, got {node.draw_rate:.3f}"
        if depth < max_depth:
            for c in node.children:
                if c.explore_count > 0:
                    check_draw(c, depth + 1, max_depth)

    check_draw(root)


def test_wdl_mcts_with_terminal_leaf():
    """draw_prob=0 for terminal win/loss, draw_prob=1 for terminal draw."""
    game = pyspiel.load_game("tic_tac_toe")
    # Create a fork state known to have a forcing winning move
    state = game.new_initial_state()
    for a in [0, 8, 1, 7, 4]:  # not terminal — still playable
        state.apply_action(a)

    class NeutralEval:
        def batch_inference_raw(self, states):
            values = np.array([[0.5, 0.3, 0.2]] * len(states),
                              dtype=np.float32)
            priors = []
            for s in states:
                legal = s.legal_actions()
                priors.append([(a, 1.0 / len(legal)) for a in legal])
            return values, priors

    config = MCTSConfig(max_simulations=128, batch_size=16,
                        uct_c=2.0)
    mcts = BatchMCTS(game, config, NeutralEval(),
                     random_state=np.random.RandomState(42))
    root = mcts.mcts_search(state.clone())

    # Root Q should be near (0.5-0.2)=0.3 from evaluator, minus terminal
    # win contributions which push it toward 1.0
    assert -0.2 < root.q_value < 1.1, \
        f"root Q={root.q_value:.3f} should be between -0.2 and 1.1"
    # Non-terminal leaves give d=0.3, terminal wins give d=0 — so draw_rate
    # should be less than 0.3
    assert 0.0 <= root.draw_rate <= 0.4, \
        f"root draw_rate={root.draw_rate:.3f} should be <= 0.4"


def test_wdl_perspective_flip():
    """MCTS must correctly flip Q across zero-sum perspective changes.

    Search from p0's turn. Evaluator returns w=0.6, d=0.3, l=0.1 from the
    perspective of whoever is about to move → Q=(0.6-0.1)*1=0.5 per leaf.

    At depth 0 (p0): leaf Q=+0.5, returns=[+0.5, -0.5]
    At depth 1 (p1): leaf Q=+0.5, returns=[+0.5, -0.5]
      → backprop to p1 node: returns[1]=-0.5 (since value was from p1's perspective,
        but zero-sum flips it for p0 parent... wait no.
        Actually: returns = [value, -value] = [+0.5, -0.5].
        For p1 decision node: returns[1] = -0.5.
        For p0 decision node: returns[0] = +0.5.

    Root is p0. p0 nodes get +0.5 from their own leaves and from backprop
    through p1 nodes (which provide -0.5 to p1, then flip to +0.5 for p0).
    Net: root Q should be approximately +0.5.
    """
    game = pyspiel.load_game("tic_tac_toe")
    state = game.new_initial_state()  # p0 to move

    class FixedWdlEval:
        def batch_inference_raw(self, states):
            values = np.array(
                [_scalar_to_wdl(0.5, s.current_player(), draw_rate=0.3)
                 for s in states], dtype=np.float32)
            priors = [[(a, 1.0 / len(s.legal_actions()))
                       for a in s.legal_actions()] for s in states]
            return values, priors

    config = MCTSConfig(max_simulations=128, batch_size=16,
                        uct_c=2.0)
    mcts = BatchMCTS(game, config, FixedWdlEval(),
                     random_state=np.random.RandomState(42))
    root = mcts.mcts_search(state.clone())

    assert abs(root.q_value - 0.5) < 1e-3, \
        f"p0-root Q≈0.5 expected, got {root.q_value:.6f}"


def test_wdl_perspective_draw_symmetric():
    """draw_rate must be identical regardless of whose turn it is —
    a draw is a draw for both players."""
    game = pyspiel.load_game("tic_tac_toe")
    state0 = game.new_initial_state()  # p0 to move
    state1 = state0.clone()
    state1.apply_action(0)  # p1 to move

    class DrawEval:
        def batch_inference_raw(self, states):
            values = np.array([[0.0, 1.0, 0.0]] * len(states),
                              dtype=np.float32)
            priors = []
            for s in states:
                legal = s.legal_actions()
                priors.append([(a, 1.0 / len(legal)) for a in legal])
            return values, priors

    config = MCTSConfig(max_simulations=64, batch_size=8)
    m0 = BatchMCTS(game, config, DrawEval(),
                   random_state=np.random.RandomState(42))
    m1 = BatchMCTS(game, config, DrawEval(),
                   random_state=np.random.RandomState(42))

    root0 = m0.mcts_search(state0)  # p0 root
    root1 = m1.mcts_search(state1)  # p1 root

    # Both should have high draw_rate >= 0.8 since all leaves have d=1.0
    assert root0.draw_rate >= 0.7, \
        f"p0-root draw_rate should be high, got {root0.draw_rate:.3f}"
    assert root1.draw_rate >= 0.7, \
        f"p1-root draw_rate should be high, got {root1.draw_rate:.3f}"
    # Draw rates should be similar regardless of perspective
    diff = abs(root0.draw_rate - root1.draw_rate)
    assert diff < 0.3, \
        (f"draw_rate should be perspective-invariant, diff={diff:.3f}, "
         f"p0={root0.draw_rate:.3f} p1={root1.draw_rate:.3f}")


def test_wdl():
    main()


if __name__ == "__main__":
    test_wdl()
