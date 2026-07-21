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
    value_learn_prob: float = 1.0  # probability of including value loss per batch
    policy_mix_alpha: float = 0.5   # advantage mixing weight
    adv_temperature: float = 0.2    # softmax temperature (lower=sharper)

    # MCTS
    max_simulations: int = 64
    mcts_batch_size: int = 32
    inference_batch_size: int = 128  # server-side max batch for shared eval
    uct_c: float = 1.41
    draw_penalty: float = 0.0   # penalise draw-heavy branches in PUCT selection
    repeat_penalty: float = 0.1  # penalise repeating positions in PUCT
    fpu_lambda: float = 0.2       # FPU: unvisited Q = Q_parent - λ·(p_max-p)/p_max
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
    num_gpus: int = 1             # self-play inference across this many GPUs
    max_steps: int = 300
    checkpoint_freq: int = 10

    # Evaluation
    evaluation_window: int = 50
    eval_reference_count: int = 5
    best_model_prob: float = 0.3       # probability of self-play vs best model
    random_opponent_prob: float = 0.2  # probability of random ckpt opponent
    eval_num_actors: int = 10          # parallel actors for eval matches
    eval_min_interval: int = 1800

    # Early termination (pruning)
    prune_enabled: bool = True
    prune_threshold: float = 0.99    # MCTS Q exceeding this triggers prune
    prune_prob: float = 0.9          # probability of actually pruning

    # Opening book
    opening_book_dir: str = ""         # directory of serialized opening states
    opening_book_prob: float = 0.0     # probability of using an opening (0=off)

    # Speculative probe (piggybacks NN eval for surprise detection)
    probe_depth: int = 0               # probe layers (0=off, 1=NN cache only)
    probe_surprise: float = 0.3        # Q-drop threshold for probe termination

    # Enhanced search: occasionally run deeper MCTS on balanced positions
    enhanced_prob: float = 0.0          # probability of triggering (0=off)
    enhanced_multiplier: int = 4        # max_simulations × multiplier
    enhanced_max_wdl: float = 0.95      # only trigger if max(w,d,l) ≤ this

    # Surprise detection (KL-based)
    surprise_pol_kl: float = 0.3       # KL(search_pol || NN_prior) threshold
    surprise_val_kl: float = 0.3       # KL(search_WDL || NN_WDL) threshold
    surprise_child_min_n: int = 150    # min visits for child to trigger

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
