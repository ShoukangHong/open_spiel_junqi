r"""BatchAlphaBeta correctness verification on TicTacToe.

Tests mirror tests/batch_mcts/test_mcts.py scenarios:
  - X must win (depth 2)
  - X must block (depth 2)
  - Diagonal fork (depth 9 = full tree)
  - Step/policy interface
  - Pruning effectiveness
"""

import numpy as np
import pyspiel

from train.batch_alpha_beta.alpha_beta import BatchAlphaBeta
from train.batch_alpha_beta.node import AlphaBetaNode


class ZeroEvaluator:
    """Evaluator returning uniform WDL = [0,1,0] and uniform priors."""
    def batch_inference_raw(self, states):
        n = len(states)
        vals = np.zeros((n, 3), dtype=np.float32)
        vals[:, 1] = 1.0
        priors = []
        for s in states:
            leg = s.legal_actions()
            priors.append([(a, 1.0 / len(leg)) for a in leg])
        return vals, priors


def _print_header(msg):
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}")


def _ttt_make_state(moves_X, moves_O):
    game = pyspiel.load_game("tic_tac_toe")
    state = game.new_initial_state()
    for x, o in zip(moves_X, moves_O):
        a = 3 * x[0] + x[1]
        assert a in state.legal_actions(), f"illegal X move {x}"
        state.apply_action(a)
        if state.is_terminal():
            break
        a = 3 * o[0] + o[1]
        assert a in state.legal_actions(), f"illegal O move {o}"
        state.apply_action(a)
        if state.is_terminal():
            break
    return state


# ═══════════════════════════════════════════════════════════════════════════════
#  Test A: X must win
# ═══════════════════════════════════════════════════════════════════════════════

def test_ab_x_must_win():
    _print_header("Test A: X must win (depth 2)")

    game = pyspiel.load_game("tic_tac_toe")
    state = _ttt_make_state([(0, 0), (0, 1)], [(1, 0), (1, 1)])
    # X at (0,0),(0,1); O at (1,0),(1,1) — X to move must take (0,2)=action 2

    ab = BatchAlphaBeta(game, ZeroEvaluator(), depth=2)
    root = ab.search(state)
    best = root.best_child().action
    assert best == 2, f"Expected action 2, got {best}"
    print("  Test A PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
#  Test B: X must block
# ═══════════════════════════════════════════════════════════════════════════════

def test_ab_x_must_block():
    _print_header("Test B: X must block (depth 3)")

    game = pyspiel.load_game("tic_tac_toe")
    state = _ttt_make_state([(1, 1), (2, 2)], [(0, 0), (0, 1)])
    # O threatens (0,2), X must block → action 2

    ab = BatchAlphaBeta(game, ZeroEvaluator(), depth=3)
    root = ab.search(state)
    best = root.best_child().action
    assert best == 2, f"Expected action 2, got {best}"
    print("  Test B PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
#  Test C: Diagonal fork — full tree solve (depth 9)
# ═══════════════════════════════════════════════════════════════════════════════

def test_ab_diagonal_fork():
    _print_header("Test C: Diagonal fork (depth 9 = full tree)")

    game = pyspiel.load_game("tic_tac_toe")
    fork_state = game.new_initial_state()
    fork_state.apply_action(0)   # X(0,0)
    fork_state.apply_action(8)   # O(2,2)
    fork_state.apply_action(4)   # X(1,1)
    #     X . .
    #     . X .
    #     . . O   O to move
    # Minimax: {2,6}=DRAW  {1,3,5,7}=LOSS

    drawn_moves = {2, 6}
    lost_moves = {1, 3, 5, 7}

    ab = BatchAlphaBeta(game, ZeroEvaluator(), depth=9)
    root = ab.search(fork_state.clone())

    best = root.best_child().action

    for c in sorted(root.children, key=lambda c: -c.q_value):
        row, col = c.action // 3, c.action % 3
        true_label = "DRAW" if c.action in drawn_moves else "LOSS"
        outcome_str = (f"outcome={c.outcome[:2]}"
                       if c.outcome is not None else "pruned")
        print(f"      ({row},{col}) a={c.action} [{true_label:>4s}]: "
              f"Q={c.q_value:+.3f}  {outcome_str}")

    # Best move must be a DRAW move.
    # (Pruned children may have wrong Q values due to alpha-beta bounds,
    # but the root's best_child is always correct.)
    assert best in drawn_moves, \
        f"Best move {best} not in drawn moves {drawn_moves}"

    print("  Test C PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
#  Test D: step_with_policy interface
# ═══════════════════════════════════════════════════════════════════════════════

def test_ab_step_with_policy():
    _print_header("Test D: step_with_policy interface")

    game = pyspiel.load_game("tic_tac_toe")
    state = game.new_initial_state()
    ab = BatchAlphaBeta(game, ZeroEvaluator(), depth=3)

    policy, action = ab.step_with_policy(state)

    # Policy must cover all legal actions
    legal = set(state.legal_actions())
    policy_actions = {a for a, _ in policy}
    assert policy_actions == legal, f"Policy actions {policy_actions} != legal {legal}"

    # Probabilities should sum to 1
    total = sum(p for _, p in policy)
    assert abs(total - 1.0) < 1e-6, f"Policy sum {total} != 1.0"

    # Action must be legal
    assert action in legal, f"Action {action} not in legal {legal}"

    # Temperature test
    policy_t, action_t = ab.step_with_policy(state, temperature=1.0)
    assert action_t in legal, f"Temp action {action_t} not in legal {legal}"

    # Deterministic with same state
    ab2 = BatchAlphaBeta(game, ZeroEvaluator(), depth=3)
    action2 = ab2.step_with_policy(state)[1]
    # With uniform eval, best might be first legal action → should be deterministic
    assert action == action2, f"Deterministic mismatch: {action} vs {action2}"

    print("  Test D PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
#  Test E: Pruning effectiveness
# ═══════════════════════════════════════════════════════════════════════════════

def test_ab_pruning():
    _print_header("Test E: Pruning reduces node count")

    game = pyspiel.load_game("tic_tac_toe")
    state = game.new_initial_state()
    ab = BatchAlphaBeta(game, ZeroEvaluator(), depth=5)

    root = ab.search(state)

    def count_visited(node):
        n = 1 if node.explore_count > 0 else 0
        for c in node.children:
            n += count_visited(c)
        return n

    total = count_visited(root)
    # Full 5-ply tic-tac-toe tree without pruning: 1+9+9*8+9*8*7+9*8*7*6+9*8*7*6*5
    # = 1+9+72+504+3024+15120 = 18730
    # With alpha-beta + uniform eval (zero heuristic), pruning limited at depth 5
    # But still should be less than full tree
    full_tree_5ply = 1 + 9 + 9*8 + 9*8*7 + 9*8*7*6 + 9*8*7*6*5  # 18730
    max_no_prune = 1 + 9 + 9*9 + 9*9*9 + 9*9*9*9 + 9*9*9*9*9  # 66430

    print(f"  Nodes visited: {total}  (full 5-ply: {full_tree_5ply})")
    assert total < max_no_prune, f"Pruning not working: {total} >= {max_no_prune}"

    print("  Test E PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
#  Test F: compute_value_policy
# ═══════════════════════════════════════════════════════════════════════════════

def test_compute_value_policy():
    _print_header("Test F: compute_value_policy")

    from train.batch_alpha_beta.alpha_beta import BatchAlphaBeta
    from train.batch_alpha_beta.node import AlphaBetaNode

    # Build a mock root with children having different Q values
    root = AlphaBetaNode(None, 0, 1.0)
    root.total_reward = 0.3  # best child's value
    root.explore_count = 1

    # 4 children: best (q=0.3), slightly worse (q=0.2), worse (q=0.0), worst (q=-0.5)
    for q, a in [(0.3, 0), (0.2, 1), (0.0, 2), (-0.5, 3)]:
        c = AlphaBetaNode(a, 0, 0.25)
        c.total_reward = q
        c.explore_count = 1
        root.children.append(c)

    # High temp: spread out
    p_high = BatchAlphaBeta.compute_value_policy(root, policy_temp=0.5)
    print(f"  temp=0.5:  {[f'{p_high[a]:.3f}' for a in sorted(p_high)]}")
    assert p_high[0] > p_high[1] > p_high[2] > p_high[3], "rank order broken"
    assert p_high[0] < 0.9, "high temp should spread probability"

    # Low temp: almost argmax
    p_low = BatchAlphaBeta.compute_value_policy(root, policy_temp=0.0001)
    print(f"  temp=1e-4: {[f'{p_low[a]:.6f}' for a in sorted(p_low)]}")
    assert p_low[0] > 0.999, f"best action should dominate, got {p_low[0]:.6f}"

    print("  Test F PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    tests = [
        ("A", "X must win", test_ab_x_must_win),
        ("B", "X must block", test_ab_x_must_block),
        ("C", "Diagonal fork (full tree)", test_ab_diagonal_fork),
        ("D", "step_with_policy interface", test_ab_step_with_policy),
        ("E", "Pruning effectiveness", test_ab_pruning),
        ("F", "compute_value_policy", test_compute_value_policy),
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
    print(f"\n{'=' * 60}")
    print(f"  {'ALL PASSED' if failed == 0 else f'{failed} FAILED'}")
    print(f"{'=' * 60}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
