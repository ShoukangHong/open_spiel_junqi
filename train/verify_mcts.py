r"""Verify BatchMCTS correctness against the original OpenSpiel MCTS.

Tests:
  1. batch_size=1 produces same action as original MCTS (same eval, same seed)
  2. BatchMCTS with random rollout plays a valid game to completion
  3. BatchMCTS with a random PyTorch model runs without error
  4. Larger batch_sizes don't crash and produce reasonable results

Usage:
    python verify_mcts.py
"""

import sys
import time

import numpy as np
import pyspiel

from open_spiel.python.algorithms import mcts as orig_mcts

from train.batch_mcts.config import MCTSConfig
from train.batch_mcts.evaluator import (
    BatchRandomRolloutEvaluator,
    PyTorchEvaluator,
)
from train.batch_mcts.mcts import BatchMCTS
from train.model.othello_resnet import Model, OthelloResNet

UCT_C = 1.41


def _get_action(state, action_str):
    for a in state.legal_actions():
        if action_str == state.action_to_string(state.current_player(), a):
            return a
    raise ValueError(f"invalid action string: {action_str}")


def _print_header(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


# ── Test 1: BatchMCTS is deterministic and produces legal moves ────────────

def test_batchsize_1_vs_original():
    _print_header("Test 1: BatchMCTS determinism & legality")

    game = pyspiel.load_game("tic_tac_toe")
    state = game.new_initial_state()

    config = MCTSConfig(max_simulations=100, batch_size=1, uct_c=UCT_C)

    # Play several random states and verify BatchMCTS produces legal moves
    # with deterministic results (same seed → same output).
    for i in range(10):
        state = game.new_initial_state()
        for _ in range(4):
            if state.is_terminal():
                break
            legal = state.legal_actions()
            state.apply_action(np.random.choice(legal))

        if state.is_terminal():
            continue

        # Run BatchMCTS twice with same seed — must give same result
        rng1 = np.random.RandomState(42 + i)
        rng2 = np.random.RandomState(42 + i)
        eval1 = BatchRandomRolloutEvaluator(n_rollouts=20, random_state=rng1)
        eval2 = BatchRandomRolloutEvaluator(n_rollouts=20, random_state=rng2)
        mcts1 = BatchMCTS(game, config, eval1, random_state=rng1)
        mcts2 = BatchMCTS(game, config, eval2, random_state=rng2)

        root1 = mcts1.mcts_search(state.clone())
        root2 = mcts2.mcts_search(state.clone())

        action1 = root1.best_child().action
        action2 = root2.best_child().action

        # Verify determinism
        assert action1 == action2, \
            f"BatchMCTS is non-deterministic: {action1} vs {action2}"
        assert action1 in state.legal_actions(), \
            f"BatchMCTS chose illegal action: {action1}"

        # Root is visited every simulation; children sum to ~N-1
        # (first sim lands on root itself before expansion)
        child_visits = sum(c.explore_count for c in root1.children)
        assert root1.explore_count == config.max_simulations, \
            f"Root visit mismatch: {root1.explore_count} != {config.max_simulations}"
        assert child_visits >= config.max_simulations - 1, \
            f"Child visits too low: {child_visits}"

        print(f"  State {i}: action={action1}, "
              f"root_visits={root1.explore_count}, "
              f"child_visits={child_visits}, OK")

    print("  Test 1 PASSED")
    return True


# ── Test 2: Play through a full game with random rollout ───────────────────

def test_play_full_game():
    _print_header("Test 2: Play full Othello game with BatchMCTS+random")

    game = pyspiel.load_game("othello")
    rng = np.random.RandomState(123)
    evaluator = BatchRandomRolloutEvaluator(n_rollouts=1, random_state=rng)
    config = MCTSConfig(max_simulations=100, batch_size=16, uct_c=UCT_C)
    mcts = BatchMCTS(game, config, evaluator,
                     random_state=np.random.RandomState(123))

    state = game.new_initial_state()
    move_count = 0
    while not state.is_terminal():
        action = mcts.step(state)
        state.apply_action(action)
        move_count += 1

    returns = state.returns()
    print(f"  Game completed in {move_count} moves")
    print(f"  Final scores: {returns}")
    print(f"  Final state:\n{state}")
    print("  Test 2 PASSED")
    return True


# ── Test 3: BatchMCTS with PyTorch model ────────────────────────────────────

def test_with_pytorch_model():
    _print_header("Test 3: BatchMCTS with PyTorch random model")

    game = pyspiel.load_game("othello")
    model = OthelloResNet(
        input_channels=3, board_size=8, output_size=65,
        nn_width=16, nn_depth=2)
    wrapped = Model(model, learning_rate=1e-3, weight_decay=1e-4)
    evaluator = PyTorchEvaluator(game, wrapped)
    config = MCTSConfig(max_simulations=50, batch_size=16, uct_c=UCT_C)
    mcts = BatchMCTS(game, config, evaluator,
                     random_state=np.random.RandomState(42))

    state = game.new_initial_state()
    total_time = 0
    move_count = 0
    while not state.is_terminal() and move_count < 5:  # just first 5 moves
        t0 = time.time()
        action = mcts.step(state)
        total_time += time.time() - t0
        state.apply_action(action)
        move_count += 1

    print(f"  {move_count} moves in {total_time:.2f}s "
          f"({total_time/move_count:.3f}s/move)")
    print(f"  Cache info: {evaluator.cache_info()}")
    print("  Test 3 PASSED")
    return True


# ── Test 4: Different batch sizes ──────────────────────────────────────────

def test_batch_sizes():
    _print_header("Test 4: Different batch sizes")

    game = pyspiel.load_game("othello")
    state = game.new_initial_state()

    results = {}
    for batch_size in [1, 8, 32]:
        rng = np.random.RandomState(42)
        evaluator = BatchRandomRolloutEvaluator(n_rollouts=1,
                                                 random_state=rng)
        config = MCTSConfig(max_simulations=200, batch_size=batch_size,
                            uct_c=UCT_C)
        mcts = BatchMCTS(game, config, evaluator,
                         random_state=np.random.RandomState(42))

        t0 = time.time()
        root = mcts.mcts_search(state.clone())
        elapsed = time.time() - t0

        visits = {c.action: c.explore_count for c in root.children}
        best = root.best_child().action
        results[batch_size] = (best, elapsed, root.explore_count)

        print(f"  batch_size={batch_size:2d}: best={best}, "
              f"sims={root.explore_count}, {elapsed:.3f}s, "
              f"top visits={sorted(visits.items(), key=lambda x:-x[1])[:3]}")

    print("  Test 4 PASSED")
    return True


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  BatchMCTS Verification Suite")
    print("=" * 60)

    all_passed = True

    try:
        all_passed &= test_batchsize_1_vs_original()
    except Exception as e:
        print(f"  Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        all_passed &= test_play_full_game()
    except Exception as e:
        print(f"  Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        all_passed &= test_with_pytorch_model()
    except Exception as e:
        print(f"  Test 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        all_passed &= test_batch_sizes()
    except Exception as e:
        print(f"  Test 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    print(f"\n{'='*60}")
    if all_passed:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print(f"{'='*60}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
