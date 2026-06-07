"""Tests for train/core/policy.py — mix_advantage."""

import numpy as np
import pyspiel
from unittest import mock

from train.core.policy import mix_advantage


def _fake_root(q_value, child_q_values):
    """Build a mock MCTS root with given Q values for children."""
    root = mock.MagicMock()
    root.q_value = q_value
    root.outcome = None
    root.explore_count = 100
    children = []
    for i, q in enumerate(child_q_values):
        c = mock.MagicMock()
        c.action = i
        c.q_value = q
        children.append(c)
    root.children = children
    return root


def _fake_state(legal_actions):
    s = mock.MagicMock()
    s.legal_actions.return_value = legal_actions
    return s


def test_mix_advantage_alpha_zero():
    """alpha=0 → output = input visit policy."""
    root = _fake_root(0.2, [0.5, 0.0, -0.3])
    state = _fake_state([0, 1, 2])
    vp = np.array([0.7, 0.2, 0.1, 0.0], dtype=np.float32)
    out = mix_advantage(vp, root, state, 4, 0.0)
    np.testing.assert_allclose(out[:3].sum(), 1.0, atol=1e-5)
    np.testing.assert_allclose(out[:3], vp[:3] / vp[:3].sum(), atol=1e-5)


def test_mix_advantage_alpha_one():
    """alpha=1 → output = softmax(advantage)."""
    root = _fake_root(0.0, [0.5, 0.0, -0.3])
    state = _fake_state([0, 1, 2])
    vp = np.array([0.3, 0.3, 0.3, 0.0], dtype=np.float32)
    out = mix_advantage(vp, root, state, 4, 1.0)
    # Advantages: [0.5, 0.0, -0.3]. Softmax: higher → more mass.
    assert out[0] > out[1] > out[2]
    assert out[3] == 0.0
    np.testing.assert_allclose(out[:3].sum(), 1.0, atol=1e-5)


def test_mix_advantage_blend():
    """alpha=0.5 → mixed between visit and softmax."""
    root = _fake_root(0.0, [0.8, -0.2])
    state = _fake_state([0, 1])
    vp = np.array([0.5, 0.5, 0.0], dtype=np.float32)
    out = mix_advantage(vp, root, state, 3, 0.5)
    # action 0 has higher advantage → softmax pushes it above 0.5
    assert out[0] > 0.5
    assert out[1] < 0.5
    np.testing.assert_allclose(out[:2].sum(), 1.0, atol=1e-5)


def test_mix_advantage_normalises():
    """Output always sums to 1 regardless of alpha."""
    for alpha, q_val in [(0.2, 0.1), (0.5, -0.3), (0.8, 0.9)]:
        root = _fake_root(q_val, [0.5, -0.2, 0.3])
        state = _fake_state([0, 1, 2])
        vp = np.array([0.4, 0.3, 0.3, 0.0], dtype=np.float32)
        out = mix_advantage(vp, root, state, 4, alpha)
        assert abs(out.sum() - 1.0) < 1e-5, f"alpha={alpha} sum={out.sum()}"


def test_mix_advantage_illegal_zero():
    """Illegal actions get zero probability."""
    root = _fake_root(0.0, [0.5, -0.5])
    state = _fake_state([0, 1])
    vp = np.array([0.6, 0.4, 0.0, 0.0], dtype=np.float32)
    out = mix_advantage(vp, root, state, 4, 0.5)
    assert out[2] == 0.0
    assert out[3] == 0.0


def test_mix_advantage_with_pyspiel():
    """Smoke test with real pyspiel state."""
    game = pyspiel.load_game("othello")
    state = game.new_initial_state()
    root = _fake_root(0.1, [0.3, -0.1, 0.4, 0.0])
    vp = np.zeros(65, dtype=np.float32)
    for a in state.legal_actions():
        vp[a] = 1.0 / len(state.legal_actions())
    out = mix_advantage(vp, root, state, 65, 0.3)
    assert abs(out.sum() - 1.0) < 1e-5
    # only legal actions get mass
    assert np.all(out[~np.asarray(state.legal_actions_mask(), dtype=bool)] == 0)


def main():
    tests = [
        test_mix_advantage_alpha_zero, test_mix_advantage_alpha_one,
        test_mix_advantage_blend, test_mix_advantage_normalises,
        test_mix_advantage_illegal_zero, test_mix_advantage_with_pyspiel,
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
