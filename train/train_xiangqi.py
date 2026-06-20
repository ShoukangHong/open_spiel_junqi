r"""Xiangqi (Chinese Chess) AlphaZero training entry point.

Usage:
    python train_xiangqi.py                  # use defaults
    python train_xiangqi.py --fresh           # start from scratch
    python train_xiangqi.py --config my.json  # custom config
"""

import argparse
import sys
import os
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from train.core.train_loop import run_training
from train.core.model_builder import build_xiangqi_model
from train.games.xiangqi.config import XiangqiTrainConfig
from train.games.xiangqi.play import play_game
from train.model.xiangqi_symmetry import XiangqiSymmetry

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Xiangqi AlphaZero training (BatchMCTS + PyTorch)")
    parser.add_argument("--config",
                        default=os.path.join(_project_root, "train", "games",
                                            "xiangqi", "config.json"))
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    run_training(
        game_name="xiangqi",
        config_class=XiangqiTrainConfig,
        build_model_fn=build_xiangqi_model,
        play_game_fn=play_game,
        SymmetryClass=XiangqiSymmetry,  # No symmetry augmentation for xiangqi (10×9 not square)
        config_path=args.config,
        fresh=args.fresh,
    )
