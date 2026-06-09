r"""Othello AlphaZero training entry point.

Usage:
    python train_othello.py                  # use defaults
    python train_othello.py --fresh           # start from scratch
    python train_othello.py --config my.json  # custom config
"""

import argparse
import sys
import os
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from train.core.train_loop import run_training
from train.core.model_builder import build_othello_model
from train.games.othello.config import OthelloTrainConfig
from train.games.othello.play import play_game
from train.model.symmetry import OthelloSymmetry

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Othello AlphaZero training (BatchMCTS + PyTorch)")
    parser.add_argument("--config",
                        default=os.path.join(_project_root, "train", "games",
                                            "othello", "config.json"))
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    run_training(
        game_name="othello",
        config_class=OthelloTrainConfig,
        build_model_fn=build_othello_model,
        play_game_fn=play_game,
        SymmetryClass=OthelloSymmetry,
        config_path=args.config,
        fresh=args.fresh,
    )
