"""Shared model builder for training and evaluation — game-agnostic."""

from train.model.model import Model


def build_model(game, cfg, NetClass, *, board_rows=None, board_cols=None):
    """Create a Model from config and a network class.

    Args:
        game: pyspiel.Game instance.
        cfg: config dataclass or dict with nn_width, nn_depth, device, etc.
        NetClass: nn.Module subclass (e.g. OthelloResNet, XiangqiResNet).
        board_rows: override spatial height (for non-square boards).
        board_cols: override spatial width (for non-square boards).

    Returns:
        Model wrapping the network.
    """
    nn_width = cfg.nn_width if hasattr(cfg, 'nn_width') else cfg.get('nn_width', 32)
    nn_depth = cfg.nn_depth if hasattr(cfg, 'nn_depth') else cfg.get('nn_depth', 6)
    device = cfg.device if hasattr(cfg, 'device') else cfg.get('device', 'cpu')
    lr = cfg.learning_rate if hasattr(cfg, 'learning_rate') else cfg.get('learning_rate', 3e-4)
    wd = cfg.weight_decay if hasattr(cfg, 'weight_decay') else cfg.get('weight_decay', 1e-4)
    ckpt_path = cfg.path if hasattr(cfg, 'path') else cfg.get('path')

    obs_shape = game.observation_tensor_shape()
    kwargs = dict(
        input_channels=obs_shape[0],
        output_size=game.num_distinct_actions(),
        nn_width=nn_width, nn_depth=nn_depth)
    if board_rows is not None and board_cols is not None:
        kwargs["board_rows"] = board_rows
        kwargs["board_cols"] = board_cols
    else:
        kwargs["board_size"] = obs_shape[1]

    net = NetClass(**kwargs)
    return Model(net, learning_rate=lr, weight_decay=wd,
                 device=device, checkpoint_path=ckpt_path)


def build_othello_model(game, cfg):
    """Convenience wrapper that builds an OthelloResNet model."""
    from train.model.othello_resnet import OthelloResNet
    return build_model(game, cfg, OthelloResNet)


def build_xiangqi_model(game, cfg):
    """Convenience wrapper that builds a XiangqiResNet model."""
    from train.model.xiangqi_resnet import XiangqiResNet
    obs_shape = game.observation_tensor_shape()
    return build_model(game, cfg, XiangqiResNet,
                       board_rows=obs_shape[1], board_cols=obs_shape[2])
