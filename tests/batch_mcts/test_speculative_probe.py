"""Test speculative probe correctness on TicTacToe."""

import numpy as np
import pyspiel

from train.batch_mcts.config import MCTSConfig
from train.batch_mcts.mcts import BatchMCTS


class FixedEval:
    """Evaluator that returns WDL=[w,d,l] for every state, uniform priors."""

    def __init__(self, w=0.4, d=0.2, l=0.4):
        self._wdl = np.array([w, d, l], dtype=np.float32)

    def batch_inference_raw(self, states):
        n = len(states)
        vals = np.tile(self._wdl, (n, 1))
        priors = []
        for s in states:
            leg = s.legal_actions()
            priors.append([(a, 1.0 / len(leg)) for a in leg])
        return vals, priors


def _print_header(msg):
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Test A: probe_depth=1, max_sim=0 — verify per-node Q values
# ═══════════════════════════════════════════════════════════════════════════════

def test_probe_level0_q():
    _print_header("Test A: probe_depth=1 correct Q per root child")

    game = pyspiel.load_game("tic_tac_toe")
    # WDL=[0.3, 0.4, 0.3] → q_nn = 0.3-0.3 = 0
    # Root player=0. Child's player=1. NN's WDL is from player 1's view.
    # q_nn = 0 (from player 1). Root perspective = -0 = 0.
    eval_draw = FixedEval(w=0.3, d=0.4, l=0.3)

    cfg = MCTSConfig(max_simulations=0, batch_size=1, probe_depth=1,
                     probe_surprise=10.0)
    mcts = BatchMCTS(game, cfg, eval_draw,
                     random_state=np.random.RandomState(42))
    state = game.new_initial_state()
    root = mcts.mcts_search(state.clone())

    print(f"  root.N={root.explore_count}  root.Q={root.q_value:.4f}")
    for c in root.children[:5]:
        print(f"    a={c.action}  N={c.explore_count}  Q={c.q_value:.4f}")

    # Each child: 1 backprop from level 0 → N=1
    for c in root.children:
        assert c.explore_count == 1, \
            f"Child {c.action} N={c.explore_count}, expected 1"
        assert abs(c.q_value - 0.0) < 0.01, \
            f"Child {c.action} Q={c.q_value:.4f}, expected 0.0"

    # Root: sum of children N
    assert root.explore_count == len(root.children), \
        f"Root N={root.explore_count}, expected {len(root.children)}"

    print("  Test A PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
#  Test B: probe_depth=1 with WDL ≠ draw — Q sign correct
# ═══════════════════════════════════════════════════════════════════════════════

def test_probe_q_sign():
    _print_header("Test B: probe Q sign correct for non-draw WDL")

    game = pyspiel.load_game("tic_tac_toe")
    # WDL=[0.8, 0.1, 0.1] → q_nn = 0.7 from evaluated state's view.
    # Root player=0 (X). Child state player=1 (O).
    # WDL is from player 1's view → player 1 has 80% win chance.
    # From root (player 0): player 1's win = player 0's loss.
    # Root Q = -(0.8-0.1) = -0.7
    eval_p1_win = FixedEval(w=0.8, d=0.1, l=0.1)

    cfg = MCTSConfig(max_simulations=0, batch_size=1, probe_depth=1,
                     probe_surprise=10.0)
    mcts = BatchMCTS(game, cfg, eval_p1_win,
                     random_state=np.random.RandomState(42))
    state = game.new_initial_state()
    root = mcts.mcts_search(state.clone())

    expected_q = -0.7  # root perspective: opponent wins → negative
    print(f"  root.Q={root.q_value:.4f}  expected={expected_q:.4f}")
    for c in root.children[:3]:
        print(f"    a={c.action}  Q={c.q_value:.4f}")

    for c in root.children:
        assert abs(c.q_value - expected_q) < 0.02, \
            f"Child {c.action} Q={c.q_value:.4f}, expected {expected_q:.4f}"

    print("  Test B PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
#  Test C: probe_depth=2 — two levels of backprop, counts correct
# ═══════════════════════════════════════════════════════════════════════════════

def test_probe_depth2_counts():
    _print_header("Test C: probe_depth=2 — N backprops per path")

    game = pyspiel.load_game("tic_tac_toe")
    eval_draw = FixedEval(w=0.3, d=0.4, l=0.3)
    pd = 2
    cfg = MCTSConfig(max_simulations=0, batch_size=1, probe_depth=pd,
                     probe_surprise=10.0)
    mcts = BatchMCTS(game, cfg, eval_draw,
                     random_state=np.random.RandomState(42))
    state = game.new_initial_state()
    root = mcts.mcts_search(state.clone())

    print(f"  root.N={root.explore_count}")
    for c in root.children[:5]:
        print(f"    a={c.action}  child.N={c.explore_count}  "
              f"Q={c.q_value:.4f}")

    # Each root child gets pd backprops
    for c in root.children:
        assert c.explore_count == pd, \
            f"Child {c.action} N={c.explore_count}, expected {pd}"

    assert root.explore_count == len(root.children) * pd, \
        f"Root N={root.explore_count}, expected {len(root.children) * pd}"

    print("  Test C PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
#  Test D: probe_depth=2 Q — only leaf value backpropped
# ═══════════════════════════════════════════════════════════════════════════════

def test_probe_depth2_q():
    _print_header("Test D: probe_depth=2 — leaf value only")

    game = pyspiel.load_game("tic_tac_toe")
    # WDL=[0.9, 0.0, 0.1] at every level
    eval_biased = FixedEval(w=0.9, d=0.0, l=0.1)
    cfg = MCTSConfig(max_simulations=0, batch_size=1, probe_depth=2,
                     probe_surprise=10.0)
    mcts = BatchMCTS(game, cfg, eval_biased,
                     random_state=np.random.RandomState(42))
    state = game.new_initial_state()
    root = mcts.mcts_search(state.clone())

    # probe_depth=2: walk level 0→1→2, backprop at depth 2 (leaf).
    # Leaf (depth 2, player=0): WDL from player 0, q=0.8.
    # Backprop: ret=[0.8, -0.8]. root gets +0.8. child gets +0.8.
    # Each child: 1 visit, Q = root perspective leaf value

    print(f"  root.Q={root.q_value:.4f}")
    for c in root.children[:3]:
        print(f"    a={c.action}  child.Q={c.q_value:.4f}")

    for c in root.children:
        assert c.explore_count == 2, f"Child {c.action} N={c.explore_count}"
        assert abs(c.q_value - 0.8) < 0.02, \
            f"Child {c.action} Q={c.q_value:.4f}, expected ~0.8"

    print("  Test D PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
#  Test E: probe with max_sim>0 preserves best action
# ═══════════════════════════════════════════════════════════════════════════════

def test_probe_with_mcts():
    _print_header("Test E: probe + MCTS preserves best action")

    game = pyspiel.load_game("tic_tac_toe")
    from train.batch_mcts.evaluator import BatchRandomRolloutEvaluator
    rng = np.random.RandomState(42)
    ev = BatchRandomRolloutEvaluator(n_rollouts=10, random_state=rng)

    cfg_p = MCTSConfig(max_simulations=256, batch_size=8, uct_c=2.0,
                       probe_depth=2, probe_surprise=10.0, solve=True)
    cfg_n = MCTSConfig(max_simulations=256, batch_size=8, uct_c=2.0,
                       probe_depth=0, solve=True)

    mcts_p = BatchMCTS(game, cfg_p, ev,
                       random_state=np.random.RandomState(42))
    mcts_n = BatchMCTS(game, cfg_n, ev,
                       random_state=np.random.RandomState(42))

    state = game.new_initial_state()
    for a in [0, 4]:
        state.apply_action(a)

    root_p = mcts_p.mcts_search(state.clone())
    root_n = mcts_n.mcts_search(state.clone())

    top_p = {c.action for c in sorted(root_p.children,
              key=lambda c: -c.explore_count)[:3]}
    top_n = {c.action for c in sorted(root_n.children,
              key=lambda c: -c.explore_count)[:3]}

    print(f"  probe best={root_p.best_child().action}"
          f"  no_probe best={root_n.best_child().action}")
    print(f"  probe Q={root_p.q_value:.3f}  no_probe Q={root_n.q_value:.3f}")
    print(f"  probe top3={top_p}  no_probe top3={top_n}"
          f"  overlap={top_p & top_n}")

    assert top_p & top_n, f"No overlap: probe={top_p} no_probe={top_n}"

    print("  Test E PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
#  Test F: deterministic — same seed, same result
# ═══════════════════════════════════════════════════════════════════════════════

def test_probe_deterministic():
    _print_header("Test F: probe + MCTS is deterministic")

    game = pyspiel.load_game("tic_tac_toe")
    from train.batch_mcts.evaluator import BatchRandomRolloutEvaluator

    cfg = MCTSConfig(max_simulations=128, batch_size=8, uct_c=2.0,
                     probe_depth=2, probe_surprise=10.0, solve=True)

    state = game.new_initial_state()
    for a in [0, 4]:
        state.apply_action(a)

    results = []
    for _ in range(3):
        # Fresh evaluator each run so random state is identical
        ev = BatchRandomRolloutEvaluator(
            n_rollouts=5, random_state=np.random.RandomState(42))
        mcts = BatchMCTS(game, cfg, ev,
                         random_state=np.random.RandomState(123))
        root = mcts.mcts_search(state.clone())
        results.append(root.best_child().action)

    print(f"  best actions across 3 runs: {results}")
    assert len(set(results)) == 1, \
        f"Non-deterministic! Got different best actions: {results}"

    print("  Test F PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
#  Test G: probe with uniform priors (stress test for tiebreaker)
# ═══════════════════════════════════════════════════════════════════════════════

def test_probe_uniform_priors():
    _print_header("Test G: probe with uniform priors — deterministic")

    game = pyspiel.load_game("tic_tac_toe")
    ev = FixedEval(w=0.3, d=0.4, l=0.3)  # uniform priors, all Q=0

    cfg = MCTSConfig(max_simulations=16, batch_size=4, uct_c=2.0,
                     probe_depth=3, probe_surprise=10.0, solve=True)

    state = game.new_initial_state()
    for a in [0, 4]:
        state.apply_action(a)

    results = []
    for _ in range(3):
        ev_fresh = FixedEval(w=0.3, d=0.4, l=0.3)
        mcts = BatchMCTS(game, cfg, ev_fresh,
                         random_state=np.random.RandomState(99))
        root = mcts.mcts_search(state.clone())
        results.append(root.best_child().action)

    print(f"  best actions: {results}")
    assert len(set(results)) == 1, \
        f"Non-deterministic with uniform priors: {results}"

    print("  Test G PASSED")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    tests = [
        ("A", "probe_depth=1 Q", test_probe_level0_q),
        ("B", "Q sign correct", test_probe_q_sign),
        ("C", "depth=2 counts", test_probe_depth2_counts),
        ("D", "depth=2 Q alternation", test_probe_depth2_q),
        ("E", "probe+MCTS agreement", test_probe_with_mcts),
        ("F", "deterministic", test_probe_deterministic),
        ("G", "uniform priors", test_probe_uniform_priors),
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
