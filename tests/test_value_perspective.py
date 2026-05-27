"""Test value perspective handling for scalar vs WDL modes.

Key invariant:
  - value_classes=1: scalar, p0 perspective (returns[0]), backward compat
  - value_classes=3: WDL one-hot, cur_player perspective (returns[cur_player])
"""

import numpy as np
import pytest
import torch

from train.core.replay_buffer import ReplayBuffer
from train.model.othello_resnet import Model, OthelloResNet, TrainInput
from train.train_othello import _make_value


# ── _make_value ─────────────────────────────────────────────────────────────

def test_make_value_scalar_p0_wins():
    """p0 wins → returns[0] = +1"""
    v = _make_value(+1.0, value_classes=1)
    assert v == 1.0
    assert isinstance(v, float)


def test_make_value_scalar_p0_loses():
    v = _make_value(-1.0, value_classes=1)
    assert v == -1.0


def test_make_value_scalar_draw():
    v = _make_value(0.0, value_classes=1)
    assert v == 0.0


def test_make_value_wdl_win():
    v = _make_value(+1.0, value_classes=3)
    np.testing.assert_array_equal(v, [1.0, 0.0, 0.0])


def test_make_value_wdl_draw():
    v = _make_value(0.0, value_classes=3)
    np.testing.assert_array_equal(v, [0.0, 1.0, 0.0])


def test_make_value_wdl_loss():
    v = _make_value(-1.0, value_classes=3)
    np.testing.assert_array_equal(v, [0.0, 0.0, 1.0])


# ── Buffer round-trip ──────────────────────────────────────────────────────

def test_buffer_scalar_roundtrip():
    buf = ReplayBuffer(100, value_dim=1)
    obs = np.zeros((4, 8, 8), dtype=np.float32)
    mask = np.ones(65, dtype=bool)
    pol = np.ones(65, dtype=np.float32) / 65
    buf.append(obs, mask, pol, 0.5)
    buf.append(obs, mask, pol, -0.3)
    batch = buf.sample(2)
    assert batch.value.shape == (2,)
    assert 0.5 in batch.value or -0.3 in batch.value


def test_buffer_wdl_roundtrip():
    buf = ReplayBuffer(100, value_dim=3)
    obs = np.zeros((4, 8, 8), dtype=np.float32)
    mask = np.ones(65, dtype=bool)
    pol = np.ones(65, dtype=np.float32) / 65
    buf.append(obs, mask, pol, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    buf.append(obs, mask, pol, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    batch = buf.sample(2)
    assert batch.value.shape == (2, 3)
    assert batch.value.sum(axis=-1).mean() == pytest.approx(1.0)


# ── Model forward ───────────────────────────────────────────────────────────

def test_model_scalar_forward():
    net = OthelloResNet(input_channels=4, board_size=8, output_size=65,
                        nn_width=8, nn_depth=1, num_value_classes=1)
    x = torch.randn(2, 4, 8, 8)
    logits, val = net(x)
    assert val.shape == (2,)
    assert val.min() >= -1.0 and val.max() <= 1.0  # tanh output


def test_model_wdl_forward():
    net = OthelloResNet(input_channels=4, board_size=8, output_size=65,
                        nn_width=8, nn_depth=1, num_value_classes=3)
    x = torch.randn(2, 4, 8, 8)
    logits, val = net(x)
    assert val.shape == (2, 3)
    # Raw logits — no softmax applied in forward()


def test_model_scalar_inference():
    net = OthelloResNet(input_channels=4, board_size=8, output_size=65,
                        nn_width=8, nn_depth=1, num_value_classes=1)
    obs = np.zeros((4, 8, 8), dtype=np.float32)
    mask = np.ones(65, dtype=bool)
    val, pol = net.inference(obs, mask)
    assert isinstance(val, float)
    assert -1.0 <= val <= 1.0


def test_model_wdl_inference():
    net = OthelloResNet(input_channels=4, board_size=8, output_size=65,
                        nn_width=8, nn_depth=1, num_value_classes=3)
    obs = np.zeros((4, 8, 8), dtype=np.float32)
    mask = np.ones(65, dtype=bool)
    val, pol = net.inference(obs, mask)
    assert isinstance(val, np.ndarray)
    assert val.shape == (3,)
    assert abs(val.sum() - 1.0) < 0.01  # softmaxed


def test_model_wdl_batch_inference():
    net = OthelloResNet(input_channels=4, board_size=8, output_size=65,
                        nn_width=8, nn_depth=1, num_value_classes=3)
    obs = np.zeros((4, 256), dtype=np.float32)
    mask = np.ones((4, 65), dtype=bool)
    vals, pols = net.batch_inference(obs, mask)
    assert vals.shape == (4, 3)
    np.testing.assert_allclose(vals.sum(axis=-1), 1.0, atol=0.01)


# ── Model.update ────────────────────────────────────────────────────────────

def test_model_update_scalar():
    net = OthelloResNet(input_channels=4, output_size=65,
                        nn_width=8, nn_depth=1, num_value_classes=1)
    model = Model(net, device="cpu")
    obs = np.random.randn(4, 256).astype(np.float32)
    mask = np.ones((4, 65), dtype=bool)
    pol = np.random.rand(4, 65).astype(np.float32)
    pol /= pol.sum(axis=-1, keepdims=True)
    val = np.array([0.5, -0.3, 0.0, 0.8], dtype=np.float32)
    batch = TrainInput(observation=obs, legals_mask=mask, policy=pol, value=val)
    loss = model.update(batch)
    assert loss.value > 0  # MSE loss is positive


def test_model_update_wdl():
    net = OthelloResNet(input_channels=4, output_size=65,
                        nn_width=8, nn_depth=1, num_value_classes=3)
    model = Model(net, device="cpu")
    obs = np.random.randn(4, 256).astype(np.float32)
    mask = np.ones((4, 65), dtype=bool)
    pol = np.random.rand(4, 65).astype(np.float32)
    pol /= pol.sum(axis=-1, keepdims=True)
    val = np.array([[1,0,0],[0,1,0],[0,0,1],[1,0,0]], dtype=np.float32)
    batch = TrainInput(observation=obs, legals_mask=mask, policy=pol, value=val)
    loss = model.update(batch)
    assert loss.value > 0  # CE loss is positive


# ── MCTS evaluator ──────────────────────────────────────────────────────────

def test_evaluator_scalar_returns_zero_sum():
    import pyspiel
    from train.batch_mcts.evaluator import PyTorchEvaluator

    game = pyspiel.load_game("othello")
    net = OthelloResNet(input_channels=4, output_size=65,
                        nn_width=8, nn_depth=1, num_value_classes=1)
    model = Model(net, device="cpu")
    ev = PyTorchEvaluator(game, model, value_classes=1)

    state = game.new_initial_state()
    ret = ev.evaluate(state)
    assert ret.shape == (2,)
    assert abs(ret[0] + ret[1]) < 0.01  # zero-sum: v0 = -v1


def test_evaluator_wdl_returns_triple():
    import pyspiel
    from train.batch_mcts.evaluator import PyTorchEvaluator

    game = pyspiel.load_game("othello")
    net = OthelloResNet(input_channels=4, output_size=65,
                        nn_width=8, nn_depth=1, num_value_classes=3)
    model = Model(net, device="cpu")
    ev = PyTorchEvaluator(game, model, value_classes=3)

    state = game.new_initial_state()
    ret = ev.evaluate(state)
    assert ret.shape == (3,)
    assert abs(ret.sum() - 1.0) < 0.01


# ── Perspective integration ─────────────────────────────────────────────────

def test_full_pipeline_scalar():
    """End-to-end: _make_value → buffer → model.update for scalar mode."""
    buf = ReplayBuffer(100, value_dim=1)
    obs = np.zeros((4, 8, 8), dtype=np.float32)
    mask = np.ones(65, dtype=bool)
    pol = np.ones(65, dtype=np.float32) / 65

    # Simulate a game where p0 wins (returns[0]=+1)
    for cur_player in [0, 1]:
        # Scalar always uses returns[0], regardless of cur_player
        v = _make_value(1.0, value_classes=1)
        assert v == 1.0  # p0 perspective
        buf.append(obs, mask, pol, v)

    net = OthelloResNet(input_channels=4, output_size=65,
                        nn_width=8, nn_depth=1, num_value_classes=1)
    model = Model(net, device="cpu")
    batch = buf.sample(4)
    loss = model.update(batch)
    assert loss.value > 0


def test_full_pipeline_wdl():
    """End-to-end: _make_value → buffer → model.update for WDL mode."""
    buf = ReplayBuffer(100, value_dim=3)
    obs = np.zeros((4, 8, 8), dtype=np.float32)
    mask = np.ones(65, dtype=bool)
    pol = np.ones(65, dtype=np.float32) / 65

    # returns[cur_player=0] = +1 (win) → [1,0,0]
    buf.append(obs, mask, pol, _make_value(+1.0, value_classes=3))
    # returns[cur_player=1] = +1 → also win for that player → [1,0,0]
    buf.append(obs, mask, pol, _make_value(+1.0, value_classes=3))
    # returns[cur_player=0] = 0 → draw → [0,1,0]
    buf.append(obs, mask, pol, _make_value(0.0, value_classes=3))
    # returns[cur_player=1] = -1 → loss for that player → [0,0,1]
    buf.append(obs, mask, pol, _make_value(-1.0, value_classes=3))

    net = OthelloResNet(input_channels=4, output_size=65,
                        nn_width=8, nn_depth=1, num_value_classes=3)
    model = Model(net, device="cpu")
    batch = buf.sample(4)
    loss = model.update(batch)
    assert loss.value > 0


# ── run ─────────────────────────────────────────────────────────────────────

def main():
    tests = [
        test_make_value_scalar_p0_wins, test_make_value_scalar_p0_loses,
        test_make_value_scalar_draw,
        test_make_value_wdl_win, test_make_value_wdl_draw, test_make_value_wdl_loss,
        test_buffer_scalar_roundtrip, test_buffer_wdl_roundtrip,
        test_model_scalar_forward, test_model_wdl_forward,
        test_model_scalar_inference, test_model_wdl_inference,
        test_model_wdl_batch_inference,
        test_model_update_scalar, test_model_update_wdl,
        test_evaluator_scalar_returns_zero_sum, test_evaluator_wdl_returns_triple,
        test_full_pipeline_scalar, test_full_pipeline_wdl,
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


def test_value_perspective():
    main()


if __name__ == "__main__":
    test_value_perspective()
