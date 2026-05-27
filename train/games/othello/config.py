"""Othello-specific training config."""

from dataclasses import dataclass
from train.core.base_config import BaseTrainConfig


@dataclass
class OthelloTrainConfig(BaseTrainConfig):
    game: str = "othello"

    # Weak-move exploration
    weak_side_prob: float = 0.5
    weak_move_prob: float = 0.5
    rare_case_threshold: float = 0.4
    weak_move_threshold: float = 0.2
    weak_max_per_game: int = 3
    weak_move_max_step: int = 100

    path: str = "othello_train_v2"
