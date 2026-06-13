"""Tests for XiangqiResNet: shapes, forward, inference, plane encoding."""

import numpy as np
import torch
import pyspiel

from train.model.xiangqi_resnet import XiangqiResNet
from train.model.model import Model
from train.core.types import TrainInput


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_net(nn_width=8, nn_depth=1):
    return XiangqiResNet(input_channels=17, board_rows=10, board_cols=9,
                         output_size=8100, nn_width=nn_width,
                         nn_depth=nn_depth)


# ── Forward pass ─────────────────────────────────────────────────────────────

def test_forward_shapes():
    """Forward returns policy_logits (batch, 8100) and value (batch, 3)."""
    net = _make_net()
    x = torch.randn(2, 17, 10, 9)
    pl, v = net(x)
    assert pl.shape == (2, 8100)
    assert v.shape == (2, 3)


def test_inference_shapes():
    """Single inference returns value (3,) and policy (8100,)."""
    net = _make_net()
    obs = np.zeros((17, 10, 9), dtype=np.float32)
    mask = np.ones(8100, dtype=bool)
    val, pol = net.inference(obs, mask)
    assert val.shape == (3,)
    assert abs(val.sum() - 1.0) < 0.01
    assert pol.shape == (8100,)
    assert abs(pol.sum() - 1.0) < 0.01


def test_batch_inference_shapes():
    """Batch inference returns values (batch, 3) and policies (batch, 8100)."""
    net = _make_net()
    obs = np.zeros((4, 1530), dtype=np.float32)  # flat input
    mask = np.ones((4, 8100), dtype=bool)
    vals, pols = net.batch_inference(obs, mask)
    assert vals.shape == (4, 3)
    np.testing.assert_allclose(vals.sum(axis=-1), 1.0, atol=0.01)
    assert pols.shape == (4, 8100)


def test_inference_respects_mask():
    """Policy is zero on masked (illegal) actions."""
    net = _make_net()
    obs = np.zeros((17, 10, 9), dtype=np.float32)
    mask = np.zeros(8100, dtype=bool)
    mask[0] = True  # only action 0 is legal
    _, pol = net.inference(obs, mask)
    assert pol[0] > 0
    assert pol[1:].sum() < 1e-6


# ── Plane encoding ───────────────────────────────────────────────────────────

def test_plane_encoding_reshape_order():
    """Reshape (90, 10, 9) → (8100,) uses C-order (plane-major).

    plane p at position (r,c) → flat index p*90 + r*9 + c,
    which matches OpenSpiel's action encoding: from_sq * 90 + to_sq.
    """
    # Build known array where plane p, row r, col c = p*1000 + r*100 + c
    planes = np.arange(90, dtype=np.float32)
    rows = np.arange(10, dtype=np.float32)
    cols = np.arange(9, dtype=np.float32)
    arr_3d = planes[:, None, None] * 1000 + rows[None, :, None] * 100 + cols[None, None, :]
    # (90, 10, 9) → C-order flatten
    flat = arr_3d.reshape(-1)

    # Verify the mapping: index = p*90 + r*9 + c
    for p in range(90):
        for r in range(10):
            for c in range(9):
                idx = p * 90 + r * 9 + c
                expected = p * 1000 + r * 100 + c
                assert flat[idx] == expected, \
                    f"Mismatch at (p={p}, r={r}, c={c}): flat[{idx}]={flat[idx]} != {expected}"


def test_action_reconstruction():
    """Verify plane encoding round-trips through pyspiel action space."""
    game = pyspiel.load_game("xiangqi")
    state = game.new_initial_state()

    # Test a few known legal actions
    legal = state.legal_actions()
    for action in legal[:5]:
        from_sq = action // 90
        to_sq = action % 90
        # Reverse mapping: action → (plane, row, col)
        plane = from_sq
        tgt_row = to_sq // 9
        tgt_col = to_sq % 9
        assert 0 <= plane < 90
        assert 0 <= tgt_row < 10
        assert 0 <= tgt_col < 9
        # And back
        reconstructed = plane * 90 + tgt_row * 9 + tgt_col
        assert reconstructed == action


# ── Model.update ─────────────────────────────────────────────────────────────

def test_model_update_wdl():
    """Model.update with WDL targets returns positive losses."""
    net = _make_net()
    model = Model(net, device="cpu")
    obs = np.random.randn(4, 1530).astype(np.float32)
    mask = np.ones((4, 8100), dtype=bool)
    pol = np.random.rand(4, 8100).astype(np.float32)
    pol /= pol.sum(axis=-1, keepdims=True)
    val = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]],
                   dtype=np.float32)
    batch = TrainInput(observation=obs, legals_mask=mask, policy=pol, value=val)
    loss = model.update(batch)
    assert loss.policy > 0
    assert loss.value > 0


# ── MCTS evaluator ───────────────────────────────────────────────────────────

def test_evaluator_wdl_returns_triple():
    from train.batch_mcts.evaluator import PyTorchEvaluator
    game = pyspiel.load_game("xiangqi")
    net = _make_net()
    model = Model(net, device="cpu")
    ev = PyTorchEvaluator(game, model)

    state = game.new_initial_state()
    ret = ev.evaluate(state)
    assert ret.shape == (3,)
    assert abs(ret.sum() - 1.0) < 0.01


def test_mcts_search_runs():
    from train.batch_mcts.config import MCTSConfig
    from train.batch_mcts.mcts import BatchMCTS
    from train.batch_mcts.evaluator import PyTorchEvaluator

    game = pyspiel.load_game("xiangqi")
    net = _make_net()
    model = Model(net, device="cpu")
    ev = PyTorchEvaluator(game, model)
    mcfg = MCTSConfig(max_simulations=8, batch_size=4)
    mcts = BatchMCTS(game, mcfg, ev, random_state=np.random.RandomState(42))

    state = game.new_initial_state()
    root = mcts.mcts_search(state)
    assert root.explore_count == 8
    assert len(root.children) >= 1
    best = root.best_child()
    assert best.action in state.legal_actions()


# ── Model builder ────────────────────────────────────────────────────────────

def test_build_xiangqi_model():
    from train.core.model_builder import build_xiangqi_model
    from train.games.xiangqi.config import XiangqiTrainConfig

    game = pyspiel.load_game("xiangqi")
    cfg = XiangqiTrainConfig()
    cfg.nn_width = 8
    cfg.nn_depth = 1
    model = build_xiangqi_model(game, cfg)
    assert model.num_trainable_variables > 0

    # Verify forward pass works
    obs = np.zeros((2, 1530), dtype=np.float32)
    mask = np.ones((2, 8100), dtype=bool)
    vals, pols = model.batch_inference(obs, mask)
    assert vals.shape == (2, 3)
    assert pols.shape == (2, 8100)


# ── play_game integration ────────────────────────────────────────────────────

def test_play_game_xiangqi():
    from train.core.play import play_game
    from train.batch_mcts.config import MCTSConfig
    from train.batch_mcts.mcts import BatchMCTS
    from train.batch_mcts.evaluator import PyTorchEvaluator
    from train.games.xiangqi.config import XiangqiTrainConfig

    game = pyspiel.load_game("xiangqi")
    net = _make_net(nn_width=4, nn_depth=0)
    model = Model(net, device="cpu")
    ev = PyTorchEvaluator(game, model)
    mcfg = MCTSConfig(max_simulations=8, batch_size=4)
    mcts = BatchMCTS(game, mcfg, ev, random_state=np.random.RandomState(42))

    cfg = XiangqiTrainConfig()
    cfg.max_simulations = 8
    cfg.mcts_batch_size = 4
    cfg.weak_max_per_game = 0
    cfg.prune_enabled = False
    cfg.policy_mix_alpha = 0

    states_info, returns, rare_games, wstats = play_game(
        game, mcts, mcts, cfg, np.random.RandomState(42), allow_weak=False)

    assert len(states_info) > 0
    assert len(returns) == 2
    assert len(rare_games) == 0
    # Each state_info entry: (obs, mask, policy, cur_player, tag, q, dr)
    for item in states_info:
        obs, mask, policy, cur_player, tag, q, dr = item
        assert obs.shape == (1530,)  # 15*10*9 flat
        assert mask.shape == (8100,)
        assert policy.shape == (8100,)
        assert abs(policy.sum() - 1.0) < 0.02
        assert cur_player in (0, 1)
        assert isinstance(q, float)
        assert isinstance(dr, float)
