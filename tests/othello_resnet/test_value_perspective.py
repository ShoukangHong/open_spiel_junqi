"""Test WDL value head, buffer round-trip, model forward/update."""

import numpy as np
import torch

from train.core.replay_buffer import ReplayBuffer
from train.model.othello_resnet import Model, OthelloResNet, TrainInput


# ── Buffer round-trip ──────────────────────────────────────────────────────

def test_buffer_wdl_roundtrip():
    buf = ReplayBuffer(100)
    obs = np.zeros((4, 8, 8), dtype=np.float32)
    mask = np.ones(65, dtype=bool)
    pol = np.ones(65, dtype=np.float32) / 65
    buf.append(obs, mask, pol, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    buf.append(obs, mask, pol, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    batch = buf.sample(2)
    assert batch.value.shape == (2, 3)
    assert abs(batch.value.sum(axis=-1).mean() - 1.0) < 0.01


# ── Model forward ───────────────────────────────────────────────────────────

def test_model_wdl_forward():
    net = OthelloResNet(input_channels=4, board_size=8, output_size=65,
                        nn_width=8, nn_depth=1)
    x = torch.randn(2, 4, 8, 8)
    logits, val = net(x)
    assert val.shape == (2, 3)  # raw WDL logits


def test_model_wdl_inference():
    net = OthelloResNet(input_channels=4, board_size=8, output_size=65,
                        nn_width=8, nn_depth=1)
    obs = np.zeros((4, 8, 8), dtype=np.float32)
    mask = np.ones(65, dtype=bool)
    val, pol = net.inference(obs, mask)
    assert val.shape == (3,)
    assert abs(val.sum() - 1.0) < 0.01  # softmaxed


def test_model_wdl_batch_inference():
    net = OthelloResNet(input_channels=4, board_size=8, output_size=65,
                        nn_width=8, nn_depth=1)
    obs = np.zeros((4, 256), dtype=np.float32)
    mask = np.ones((4, 65), dtype=bool)
    vals, pols = net.batch_inference(obs, mask)
    assert vals.shape == (4, 3)
    np.testing.assert_allclose(vals.sum(axis=-1), 1.0, atol=0.01)


# ── Model.update ────────────────────────────────────────────────────────────

def test_model_update_wdl():
    net = OthelloResNet(input_channels=4, output_size=65,
                        nn_width=8, nn_depth=1)
    model = Model(net, device="cpu")
    obs = np.random.randn(4, 256).astype(np.float32)
    mask = np.ones((4, 65), dtype=bool)
    pol = np.random.rand(4, 65).astype(np.float32)
    pol /= pol.sum(axis=-1, keepdims=True)
    val = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]],
                   dtype=np.float32)
    batch = TrainInput(observation=obs, legals_mask=mask, policy=pol, value=val)
    loss = model.update(batch)
    assert loss.value > 0  # CE loss is positive


# ── MCTS evaluator ──────────────────────────────────────────────────────────

def test_evaluator_wdl_returns_triple():
    import pyspiel
    from train.batch_mcts.evaluator import PyTorchEvaluator

    game = pyspiel.load_game("othello")
    net = OthelloResNet(input_channels=4, output_size=65,
                        nn_width=8, nn_depth=1)
    model = Model(net, device="cpu")
    ev = PyTorchEvaluator(game, model)

    state = game.new_initial_state()
    ret = ev.evaluate(state)
    assert ret.shape == (3,)  # [w, d, l]
    assert abs(ret.sum() - 1.0) < 0.01


# ── Perspective: scalar_value always p0 view ────────────────────────────────

def test_scalar_value_perspective():
    """scalar_value returns p0-perspective scalar from WDL output."""
    import pyspiel
    from train.batch_mcts.evaluator import PyTorchEvaluator

    game = pyspiel.load_game("othello")
    net = OthelloResNet(input_channels=4, output_size=65,
                        nn_width=8, nn_depth=1)
    model = Model(net, device="cpu")
    ev = PyTorchEvaluator(game, model)

    state = game.new_initial_state()  # p0 to move
    q = ev.scalar_value(state)
    assert -1.0 <= q <= 1.0


# ── run ─────────────────────────────────────────────────────────────────────

def main():
    tests = [
        test_buffer_wdl_roundtrip,
        test_model_wdl_forward, test_model_wdl_inference,
        test_model_wdl_batch_inference,
        test_model_update_wdl,
        test_evaluator_wdl_returns_triple,
        test_scalar_value_perspective,
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
