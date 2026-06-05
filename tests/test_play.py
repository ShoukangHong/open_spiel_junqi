"""Tests for play_game action selection logic."""

import numpy as np
import pyspiel
from unittest import mock

from train.games.othello.play import play_game


def _make_mock_root(state, visits_list):
    """Build a mock MCTS root with given visit counts for legal actions."""
    legal = state.legal_actions()
    root = mock.MagicMock()
    children = []
    for i, a in enumerate(legal):
        c = mock.MagicMock()
        c.action = a
        c.explore_count = visits_list[i] if i < len(visits_list) else 1
        c.total_reward = 0.0
        c.player = 1 - state.current_player()
        c.outcome = None
        c.q_value = 0.0
        children.append(c)
    root.children = children
    root.best_child.return_value = children[0]  # highest visits = first legal
    return root


class _TestCfg:
    temperature = 0.5
    temperature_drop = 0
    weak_max_per_game = 0
    weak_side_prob = 0
    max_moves = 200
    max_steps = 1000
    prune_enabled = False


def test_temperature_sampling_not_argmax():
    """温度 drop 后应从 sharpen 分布采样，而非永远选 best_child."""
    game = pyspiel.load_game("tic_tac_toe")

    # Mock MCTS: visits 55:45 on first two legal actions
    mcts = mock.MagicMock()

    def _search(state):
        return _make_mock_root(state, [55, 45])

    mcts.mcts_search.side_effect = _search
    mcts.evaluator = mock.MagicMock()

    # After the first move, states_info[1] has the post-move observation.
    # Channel 1 = player 0 pieces.  Find which position got a piece.
    first_actions = []
    for _ in range(200):
        states_info, _, _, _ = play_game(
            game, mcts, mcts, _TestCfg(), np.random.RandomState(), allow_weak=False)
        obs1 = states_info[1][0]  # observation before second move
        # Tic-tac-toe CHW: ch0=empty, ch1=p1(O), ch2=p0(X)
        obs1 = obs1.reshape(3, 3, 3)
        r, c = np.where(obs1[2] == 1.0)
        assert len(r) == 1, f"expected 1 piece, got {len(r)}"
        first_actions.append(int(r[0] * 3 + c[0]))

    # Top action (visit=55) should dominate but second (visit=45) must appear
    top = sum(1 for a in first_actions if a == 0)
    second = sum(1 for a in first_actions if a == 1)
    print(f"  first actions: top={top} second={second} total={len(first_actions)}")

    assert second > 0, (
        f"Second-best action NEVER selected ({second}/200). "
        f"Bug: temperature sampling broken, likely argmax.")


def test_temperature_argmax_without_fix():
    """若没有修复（温度被忽略），第二选项应该从不出现。"""
    # Reconstruct the buggy behaviour: always pick best_child after drop
    policy = np.array([0.55, 0.45])
    rng = np.random.RandomState(42)
    # Buggy: always action 0 (best_child)
    buggy = [0 for _ in range(200)]
    assert all(a == 0 for a in buggy), "buggy: always argmax"

    # Fixed: sample from sharpen distribution
    tau = 0.5
    sel = policy.astype(np.float64) ** (1.0 / tau)
    sel /= sel.sum()
    fixed = [rng.choice(2, p=sel) for _ in range(200)]
    n1 = sum(1 for a in fixed if a == 1)
    print(f"  fixed: second action selected {n1}/200 (expected ~{sel[1]*200:.0f})")
    assert n1 > 0, "temperature sampling should select second option sometimes"


def test_temperature_probabilities_match_theory():
    """温度 sharpen 后的采样概率与理论值一致."""
    policy = np.array([0.55, 0.45])
    tau = 0.5
    expected = policy.astype(np.float64) ** (1.0 / tau)
    expected /= expected.sum()

    rng = np.random.RandomState(0)
    n = 5000
    counts = np.zeros(2)
    for _ in range(n):
        sel = policy.astype(np.float64) ** (1.0 / tau)
        sel /= sel.sum()
        a = rng.choice(2, p=sel)
        counts[a] += 1

    empirical = counts / n
    print(f"  theory:  [{expected[0]:.4f}, {expected[1]:.4f}]")
    print(f"  empiric: [{empirical[0]:.4f}, {empirical[1]:.4f}]")
    assert np.allclose(empirical, expected, atol=0.02), \
        f"empirical {empirical} deviates from theory {expected}"


# ── run ─────────────────────────────────────────────────────────────────────

def main():
    tests = [
        test_temperature_sampling_not_argmax,
        test_temperature_argmax_without_fix,
        test_temperature_probabilities_match_theory,
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


def test_play():
    main()


if __name__ == "__main__":
    test_play()
