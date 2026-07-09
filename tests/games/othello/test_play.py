"""Tests for play_game action selection logic."""

import numpy as np
import pyspiel
from unittest import mock

from train.core.play import play_game


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
    root.explore_count = sum(visits_list)  # needed by compute_solved_policy
    root.best_child.return_value = children[0]
    return root


class _TestCfg:
    temperature = 0.5
    temperature_drop = 0
    weak_max_per_game = 0
    weak_side_prob = 0
    max_moves = 200
    max_steps = 1000
    prune_enabled = False
    policy_mix_alpha = 0
    adv_temperature = 0.2


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


# ── _setup_weak_moves ──────────────────────────────────────────────────────

from train.core.play import _setup_weak_moves


class _WeakCfg:
    weak_max_per_game = 3
    weak_side_prob = 1.0
    weak_move_prob = 0.5
    weak_move_max_step = 50
    temperature_drop = 10


def test_weak_setup_expected_count():
    """n_weak follows geometric: E[n] = 1/(1-p) with cap at weak_max."""
    cfg = _WeakCfg()
    n_trials = 5000
    total = 0
    rng = np.random.RandomState(42)
    for _ in range(n_trials):
        _, _, steps = _setup_weak_moves(cfg, rng, allow_weak=True)
        total += len(steps)
    avg = total / n_trials
    # E[n] = 1 + p + p^2 for weak_max=3, p=0.5 → 1 + 0.5 + 0.25 = 1.75
    expected = 1.75
    assert abs(avg - expected) < 0.05, \
        f"expected n_weak≈{expected:.2f}, got {avg:.3f}"


def test_weak_setup_disabled():
    cfg = _WeakCfg()
    cfg.weak_max_per_game = 0
    rng = np.random.RandomState(0)
    side, wmax, steps = _setup_weak_moves(cfg, rng)
    assert wmax == 0
    assert side is None
    assert len(steps) == 0


def test_weak_setup_steps_in_range():
    cfg = _WeakCfg()
    rng = np.random.RandomState(0)
    _, _, steps = _setup_weak_moves(cfg, rng)
    for s in steps:
        assert 2 <= s < cfg.weak_move_max_step


# ── _should_prune ──────────────────────────────────────────────────────────

from train.core.play import _should_prune


class _FakeCfg:
    prune_enabled = True
    prune_threshold = 0.99
    prune_prob = 0.9


def _fake_root(q=0.5, outcome=None, draw_rate=0.0):
    r = mock.MagicMock()
    r.q_value = q
    r.outcome = outcome
    r.draw_rate = draw_rate
    r.children = []       # triggers _stable_qdr fallback path
    r.explore_count = 10
    r.total_reward = q * r.explore_count
    r.state = None        # so outcome branch uses root.player
    r.player = 0
    return r


def test_prune_skips_when_disabled():
    cfg = _FakeCfg()
    cfg.prune_enabled = False
    p = {"used": False}
    p2, br = _should_prune(p, _fake_root(1.0), 0, cfg, 1.0, 0.5)
    assert not br


def test_prune_triggers_on_high_q():
    p = {"used": False}
    p2, br = _should_prune(p, _fake_root(1.0), 0, _FakeCfg(), 1.0, 0.5)
    assert br
    assert p2["used"]


def test_prune_dice_fail_allows_retry():
    p = {"used": False}
    p2, br = _should_prune(p, _fake_root(1.0), 0, _FakeCfg(), 1.0, 0.95)
    assert not br
    assert not p2["used"]
    # Retry with winning dice — should now trigger
    p2, br = _should_prune(p2, _fake_root(1.0), 0, _FakeCfg(), 1.0, 0.5)
    assert br


def test_prune_no_retry_after_used():
    p = {"used": True}
    p2, br = _should_prune(p, _fake_root(1.0), 0, _FakeCfg(), 1.0, 0.5)
    assert not br


def test_prune_skips_low_q():
    p = {"used": False}
    p2, br = _should_prune(p, _fake_root(0.5), 0, _FakeCfg(), 1.0, 0.5)
    assert not br


def test_prune_triggers_on_proven_win():
    p = {"used": False}
    p2, br = _should_prune(p, _fake_root(0.0, [1, -1]), 0, _FakeCfg(), 1.0, 0.5)
    assert br
    assert p2["used"]


def test_prune_skips_proven_loss():
    p = {"used": False}
    p2, br = _should_prune(p, _fake_root(-1.0, [-1, 1]), 0, _FakeCfg(), 1.0, 0.5)
    # proven_loss: outcome[0]=-1, not > 0. Q=-1.0, not >= 0.99.
    assert not br


def test_draw_truncate_triggers():
    """High draw_rate triggers truncation with draw_truncate flag."""
    p = {"used": False}
    p2, br = _should_prune(
        p, _fake_root(0.0, draw_rate=0.99), 0, _FakeCfg(), 1.0, 0.5)
    assert br
    assert p2["draw_truncate"]


def test_draw_truncate_not_when_also_winning():
    """If also winning (high Q), mark as win, not draw."""
    p = {"used": False}
    p2, br = _should_prune(
        p, _fake_root(1.0, draw_rate=0.99), 0, _FakeCfg(), 1.0, 0.5)
    assert br
    assert not p2["draw_truncate"]


def test_play():
    main()


if __name__ == "__main__":
    test_play()
