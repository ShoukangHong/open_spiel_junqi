"""Base config for AlphaZero-style training — game-agnostic parameters."""

import os
from dataclasses import dataclass

# WDL value head: 3 output classes (win, draw, loss)
VALUE_CLASSES = 3


@dataclass
class BaseTrainConfig:
    # Model
    nn_width: int = 32
    nn_depth: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    train_batch_size: int = 128
    entropy_weight: float = 0.01   # policy entropy bonus (0=off)
    policy_mix_alpha: float = 0.5   # advantage mixing weight
    adv_temperature: float = 0.2    # softmax temperature (lower=sharper)

    # MCTS
    max_simulations: int = 64
    mcts_batch_size: int = 32
    inference_batch_size: int = 128  # server-side max batch for shared eval
    uct_c: float = 1.41
    draw_penalty: float = 0.0   # penalise draw-heavy branches in PUCT selection
    policy_epsilon: float = 0.25
    policy_alpha: float = 1.0
    temperature: float = 0.01
    temperature_drop: int = 30

    # Replay buffer
    replay_buffer_size: int = 50000
    buffer_sampling_frac: float = 0.1
    symmetry: int = 1

    # Training loop
    num_actors: int = 1
    max_steps: int = 300
    checkpoint_freq: int = 10

    # Evaluation
    evaluation_window: int = 50
    eval_reference_count: int = 5
    best_model_prob: float = 0.3       # probability of self-play vs best model
    eval_min_interval: int = 1800

    # Early termination (pruning)
    prune_enabled: bool = True
    prune_threshold: float = 0.99    # MCTS Q exceeding this triggers prune
    prune_prob: float = 0.9          # probability of actually pruning

    # Misc
    path: str = "train_output"
    seed: int = 42
    device: str = "cpu"

    def __post_init__(self):
        if not os.path.isabs(self.path):
            self.path = os.path.join(os.getcwd(), self.path)


def load_base_config(path: str, defaults: type) -> object:
    """Load config from JSON, layered on defaults."""
    import json
    cfg = defaults()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        print(f"[config] Loaded {path}")
    else:
        print(f"[config] {path} not found, using defaults")
    return cfg
