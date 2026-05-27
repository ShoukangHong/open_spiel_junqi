"""Shared model builder for training and evaluation."""

from train.model.othello_resnet import Model, OthelloResNet


def build_othello_model(game, cfg):
    """Create an OthelloResNet Model from a config (dataclass or dict)."""
    nn_width = cfg.nn_width if hasattr(cfg, 'nn_width') else cfg.get('nn_width', 32)
    nn_depth = cfg.nn_depth if hasattr(cfg, 'nn_depth') else cfg.get('nn_depth', 6)
    vc = cfg.value_classes if hasattr(cfg, 'value_classes') else cfg.get('value_classes', 1)
    device = cfg.device if hasattr(cfg, 'device') else cfg.get('device', 'cpu')
    lr = cfg.learning_rate if hasattr(cfg, 'learning_rate') else cfg.get('learning_rate', 3e-4)
    wd = cfg.weight_decay if hasattr(cfg, 'weight_decay') else cfg.get('weight_decay', 1e-4)
    ckpt_path = cfg.path if hasattr(cfg, 'path') else cfg.get('path')

    obs_shape = game.observation_tensor_shape()
    net = OthelloResNet(
        input_channels=obs_shape[0], board_size=obs_shape[1],
        output_size=game.num_distinct_actions(),
        nn_width=nn_width, nn_depth=nn_depth,
        num_value_classes=vc)
    return Model(net, learning_rate=lr, weight_decay=wd,
                 device=device, checkpoint_path=ckpt_path)
