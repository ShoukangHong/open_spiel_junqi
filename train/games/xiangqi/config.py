"""Xiangqi-specific training config."""

from dataclasses import dataclass
from train.core.base_config import BaseTrainConfig


@dataclass
class XiangqiTrainConfig(BaseTrainConfig):
    game: str = "xiangqi"

    # Game-specific tuning
    temperature_drop: int = 25     # xiangqi games are long (~100-300 moves)
    max_simulations: int = 128
    mcts_batch_size: int = 8
    draw_penalty: float = 0.05  # penalise draw-heavy branches in PUCT

    # Weak-move exploration
    weak_side_prob: float = 0.0
    weak_move_prob: float = 0.0
    rare_case_threshold: float = 0.0
    weak_move_threshold: float = 0.0
    weak_max_per_game: int = 3
    weak_move_max_step: int = 200   # longer games allow more weak slots

    # Path
    path: str = "xiangqi_train"
