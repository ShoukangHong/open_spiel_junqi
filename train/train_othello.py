r"""Othello AlphaZero training with BatchMCTS + PyTorch.

Usage:
    python train_othello.py                  # use defaults
    python train_othello.py --fresh           # start from scratch
    python train_othello.py --config my.json  # custom config

Architecture:
  - Single-process synchronous self-play + training
  - BatchMCTS for efficient search (batch_size=32 leaf eval)
  - PyTorch ResNet model
  - Simple replay buffer (numpy ring buffer)
  - Regular checkpointing and evaluation
"""

import argparse
import json
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
from dataclasses import asdict, dataclass
from typing import Optional

# Make sure the project root is on sys.path so that `train.*` imports work
# regardless of where the script is invoked from.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pyspiel

from train.batch_mcts.config import MCTSConfig
from train.batch_mcts.evaluator import PyTorchEvaluator
from train.batch_mcts.mcts import BatchMCTS
from train.model.othello_resnet import Losses, Model, OthelloResNet, TrainInput

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class _Tee:
    """Writes to a file and the original stdout simultaneously."""
    def __init__(self, filepath):
        self.file = open(filepath, "w", encoding="utf-8", buffering=1)
        self.stdout = sys.stdout

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


# ── Config ──────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Game
    game: str = "othello"

    # Model
    nn_width: int = 24
    nn_depth: int = 6

    # Optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    train_batch_size: int = 128

    # MCTS
    max_simulations: int = 64          # simulations per move
    mcts_batch_size: int = 32          # leaf evaluation batch size
    uct_c: float = 1.41
    policy_epsilon: float = 0.25       # Dirichlet noise weight
    policy_alpha: float = 1.0          # Dirichlet concentration
    # Temperature: temp=1 for first N moves (exploration), then switches
    # to `temperature` (near-greedy).  AlphaGo Zero: τ=1 for 30 moves, then τ→0.
    temperature: float = 0.01         # temperature after drop (near argmax)
    temperature_drop: int = 30        # moves before switching temperature

    # Replay buffer: large capacity + continuous random sampling.
    # Each step collects buffer_sampling_frac * buffer_size new states,
    # then trains the same number of random mini-batches (not full-buffer).
    replay_buffer_size: int = 50000
    buffer_sampling_frac: float = 0.1  # fraction of buffer sampled per step

    # Training loop
    num_actors: int = 1            # 1 = single-process; >1 = multi-process
    max_steps: int = 300
    checkpoint_freq: int = 10

    # Evaluation
    eval_levels: int = 3               # number of MCTS difficulty levels
    evaluation_window: int = 50        # games per eval level

    # Misc
    path: str = "othello_train_v2"
    seed: int = 42
    device: str = "cpu"

    def __post_init__(self):
        if not os.path.isabs(self.path):
            self.path = os.path.join(_SCRIPT_DIR, self.path)


def _default_config() -> TrainConfig:
    return TrainConfig()


def load_config(path: str) -> TrainConfig:
    """Load TrainConfig from JSON file, overlaid on defaults."""
    cfg = _default_config()
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


# ── Replay Buffer ───────────────────────────────────────────────────────────

class ReplayBuffer:
    """Fixed-size FIFO ring buffer for (obs, mask, policy, value) tuples."""

    def __init__(self, max_size: int):
        self._max_size = max_size
        self._obs = None       # (max_size, 3, 8, 8)
        self._masks = None     # (max_size, 65)
        self._policies = None  # (max_size, 65)
        self._values = None    # (max_size,)
        self._index = 0
        self._size = 0

    def append(self, obs: np.ndarray, mask: np.ndarray,
               policy: np.ndarray, value: float):
        """Add one training sample."""
        if self._obs is None:
            # Lazy init — infer shapes from first sample
            self._obs = np.empty((self._max_size, *obs.shape), dtype=np.float32)
            self._masks = np.empty((self._max_size, *mask.shape), dtype=np.bool)
            self._policies = np.empty((self._max_size, *policy.shape),
                                       dtype=np.float32)
            self._values = np.empty((self._max_size,), dtype=np.float32)

        idx = self._index % self._max_size
        self._obs[idx] = obs.astype(np.float32)
        self._masks[idx] = mask
        self._policies[idx] = policy.astype(np.float32)
        self._values[idx] = float(value)
        self._index += 1
        self._size = min(self._size + 1, self._max_size)

    def sample(self, n: int):
        """Sample a batch of n transitions uniformly."""
        indices = np.random.randint(0, self._size, size=n)
        return TrainInput(
            observation=self._obs[indices],
            legals_mask=self._masks[indices],
            policy=self._policies[indices],
            value=self._values[indices],
        )

    def save(self, filepath: str):
        """Serialize buffer to a .npz file."""
        if self._obs is None:
            return  # nothing to save
        np.savez_compressed(
            filepath,
            obs=self._obs[:self._size],
            masks=self._masks[:self._size],
            policies=self._policies[:self._size],
            values=self._values[:self._size],
            index=self._index,
            size=self._size,
        )

    def load(self, filepath: str):
        """Restore buffer from a .npz file."""
        data = np.load(filepath)
        self._size = int(data["size"])
        self._index = int(data["index"])
        self._max_size = max(self._max_size, self._size)
        # Re-allocate to max_size (ring buffer may grow)
        obs_shape = data["obs"].shape[1:]
        mask_shape = data["masks"].shape[1:]
        policy_shape = data["policies"].shape[1:]
        self._obs = np.empty((self._max_size, *obs_shape), dtype=np.float32)
        self._masks = np.empty((self._max_size, *mask_shape), dtype=np.bool)
        self._policies = np.empty((self._max_size, *policy_shape), dtype=np.float32)
        self._values = np.empty((self._max_size,), dtype=np.float32)
        self._obs[:self._size] = data["obs"]
        self._masks[:self._size] = data["masks"]
        self._policies[:self._size] = data["policies"]
        self._values[:self._size] = data["values"]

    def __len__(self) -> int:
        return self._size

    @property
    def total_seen(self) -> int:
        return self._index


# ── Self-play ───────────────────────────────────────────────────────────────

def play_game(game, mcts, config, rng, record=False):
    """Play one self-play game using BatchMCTS.

    Returns a list of (obs, mask, policy, current_player) tuples and
    the final game returns.  If *record* is True, also returns a list of
    per-move strings for detailed logging.
    """
    states_info = []
    trace = [] if record else None
    state = game.new_initial_state()
    move_num = 0

    while not state.is_terminal():
        cur_player = state.current_player()
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes)
            action = rng.choice(action_list, p=prob_list)
            state.apply_action(action)
            continue

        root = mcts.mcts_search(state)

        # Build policy from visit counts
        policy = np.zeros(game.num_distinct_actions(), dtype=np.float32)
        for c in root.children:
            policy[c.action] = c.explore_count

        # Temperature: τ=1 before drop (exploration), τ=config.temperature
        # after drop (near-greedy).  Policy target and action selection
        # use the same temperature — no mismatch.
        after_drop = move_num >= config.temperature_drop
        tau = config.temperature if after_drop else 1.0

        # Normalize to probabilities first, then apply temperature.
        # This avoids overflow when 1/tau is large (e.g. tau=0.01 → exp=100).
        p_sum = policy.sum()
        if p_sum > 0:
            policy = policy / p_sum
            if tau > 0 and tau != 1.0:
                policy = policy ** (1.0 / tau)
                policy /= policy.sum()
        else:
            for a in state.legal_actions():
                policy[a] = 1.0
            policy /= policy.sum()

        # Store state info before applying action
        obs = np.asarray(state.observation_tensor(), dtype=np.float32)
        mask = np.asarray(state.legal_actions_mask(), dtype=np.bool)
        states_info.append((obs, mask, policy, cur_player))

        # Sample action
        if after_drop:
            action = root.best_child().action
        else:
            action = rng.choice(len(policy), p=policy)

        # Record trace
        if record:
            player_name = "BLACK(p0)" if cur_player == 0 else "WHITE(p1)"
            action_str = state.action_to_string(cur_player, action)
            # Top-5 MCTS stats
            children_sorted = sorted(
                root.children, key=lambda c: c.explore_count, reverse=True)
            top5 = []
            for c in children_sorted[:5]:
                a_str = state.action_to_string(cur_player, c.action)
                top5.append(f"{a_str}(N={c.explore_count} Q={c.q_value:+.3f})")
            mcts_info = " | ".join(top5)

            trace.append(
                f"\n── Move {move_num + 1}  {player_name}  tau={tau:.2f} ──\n"
                f"{state}\n"
                f"MCTS top-5: {mcts_info}\n"
                f"Chosen: {action_str}"
            )

        state.apply_action(action)
        move_num += 1

    returns = state.returns()
    if record:
        trace.append(f"\n── Final (move {move_num}) ──\n{state}\n"
                     f"Returns: {returns[0]:+.0f}/{returns[1]:+.0f}")
        return states_info, returns, trace
    return states_info, returns


# ── Multi-actor support ─────────────────────────────────────────────────────

def build_model(game, cfg):
    """Create a fresh Model from config (used by actors and learner)."""
    obs_shape = game.observation_tensor_shape()
    num_actions = game.num_distinct_actions()
    net = OthelloResNet(
        input_channels=obs_shape[0], board_size=obs_shape[1],
        output_size=num_actions,
        nn_width=cfg.nn_width, nn_depth=cfg.nn_depth)
    return Model(net, learning_rate=cfg.learning_rate,
                 weight_decay=cfg.weight_decay, device=cfg.device,
                 checkpoint_path=cfg.path)


def actor_process(cfg_dict: dict, result_queue: mp.Queue):
    """Subprocess: play self-play games, send trajectories to learner."""
    # Reconstruct config (avoids pickling issues on Windows)
    cfg = TrainConfig(**cfg_dict)
    game = pyspiel.load_game(cfg.game)
    model = build_model(game, cfg)
    mcts_cfg = MCTSConfig(
        max_simulations=cfg.max_simulations, batch_size=cfg.mcts_batch_size,
        uct_c=cfg.uct_c, policy_epsilon=cfg.policy_epsilon,
        policy_alpha=cfg.policy_alpha, verbose=False)

    evaluator = PyTorchEvaluator(game, model)
    mcts = BatchMCTS(game, mcts_cfg, evaluator,
                     random_state=np.random.RandomState())
    rng = np.random.RandomState()

    loaded_step = 0
    _LATEST = -999  # sentinel for "latest" checkpoint

    while True:
        # Check for new model weights
        latest_path = os.path.join(cfg.path, f"checkpoint-{_LATEST}.pt")
        if os.path.exists(latest_path):
            mtime = os.path.getmtime(latest_path)
            if mtime > loaded_step:
                model.load_checkpoint(_LATEST)
                evaluator.clear_cache()
                loaded_step = mtime

        states_info, returns = play_game(game, mcts, cfg, rng)

        try:
            result_queue.put((states_info, returns), timeout=1)
        except queue.Full:
            print("[actor] WARNING: queue full, dropping game", flush=True)


# ── Evaluation ──────────────────────────────────────────────────────────────

def evaluate(game, model, eval_config, num_games=50):
    """Evaluate latest model vs random-rollout MCTS at multiple levels."""
    results = {}
    for level in range(eval_config.eval_levels):
        mcts_sim = int(eval_config.max_simulations * (10 ** (level / 2)))
        eval_cfg = MCTSConfig(
            max_simulations=mcts_sim,
            batch_size=eval_config.mcts_batch_size,
            uct_c=eval_config.uct_c,
            policy_epsilon=0,           # no noise during evaluation
            verbose=False,
        )
        evaluator = PyTorchEvaluator(game, model)
        mcts_bot = BatchMCTS(game, eval_cfg, evaluator,
                             random_state=np.random.RandomState())

        wins = 0
        losses = 0
        draws = 0
        rng = np.random.RandomState()

        for i in range(num_games):
            state = game.new_initial_state()
            az_player = i % 2  # alternate sides

            while not state.is_terminal():
                cur = state.current_player()
                if cur == az_player:
                    action = mcts_bot.step(state)
                else:
                    action = rng.choice(state.legal_actions())
                state.apply_action(action)

            r = state.returns()[az_player]
            if r > 0:
                wins += 1
            elif r < 0:
                losses += 1
            else:
                draws += 1

        results[level] = {"wins": wins, "losses": losses, "draws": draws,
                          "sims": mcts_sim}
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def _run_eval(game, model, cfg, step):
    """Background evaluation thread — runs matches and prints results."""
    eval_results = evaluate(game, model, cfg,
                            num_games=cfg.evaluation_window)
    for level, r in eval_results.items():
        total = r["wins"] + r["losses"] + r["draws"]
        wr = r["wins"] / max(total, 1)
        print(f"    [eval] level={level} (sims={r['sims']}): "
              f"W={r['wins']} L={r['losses']} D={r['draws']} "
              f"WR={wr:.2%}")


def main():
    parser = argparse.ArgumentParser(
        description="Othello AlphaZero training (BatchMCTS + PyTorch)")
    parser.add_argument("--config", default="train_othello_config.json",
                        help="Path to JSON config file")
    parser.add_argument("--fresh", action="store_true",
                        help="Start fresh (ignore checkpoint)")
    args = parser.parse_args()

    # Load config
    cfg = _default_config()
    if args.config:
        cfg = load_config(args.config)

    os.makedirs(cfg.path, exist_ok=True)

    # Tee stdout to a log file in the checkpoint directory.
    _tee = _Tee(os.path.join(cfg.path, "train.log"))
    sys.stdout = _tee

    # Save merged config
    with open(os.path.join(cfg.path, "train_config.json"), "w") as f:
        json.dump({k: v for k, v in cfg.__dict__.items()
                   if not k.startswith("_")}, f, indent=2, default=str)

    print(f"[train] game={cfg.game}  nn_width={cfg.nn_width}"
          f"  nn_depth={cfg.nn_depth}")
    print(f"[train] max_sim={cfg.max_simulations}"
          f"  mcts_batch={cfg.mcts_batch_size}  buffer={cfg.replay_buffer_size}")
    print(f"[train] max_steps={cfg.max_steps}  ckpt_freq={cfg.checkpoint_freq}")
    print(f"[train] path={cfg.path}")

    # Init game
    game = pyspiel.load_game(cfg.game)
    obs_shape = game.observation_tensor_shape()
    num_actions = game.num_distinct_actions()
    print(f"[train] obs_shape={obs_shape}  num_actions={num_actions}")

    # Init model
    model = build_model(game, cfg)
    print(f"[train] Model params: {model.num_trainable_variables}")

    # Replay buffer
    buffer = ReplayBuffer(max_size=cfg.replay_buffer_size)
    samples_per_step = max(
        int(cfg.replay_buffer_size * cfg.buffer_sampling_frac),
        cfg.train_batch_size)
    n_updates = samples_per_step // cfg.train_batch_size

    # Resume from checkpoint
    start_step = 0
    if not args.fresh:
        for name in os.listdir(cfg.path):
            if name.startswith("checkpoint-") and name.endswith(".pt"):
                try:
                    s = int(name.split("-")[1].split(".")[0])
                    if s > start_step:
                        start_step = s
                except ValueError:
                    pass
        if start_step > 0:
            model.load_checkpoint(start_step)
            buf_path = os.path.join(
                cfg.path, f"buffer-checkpoint-{start_step}.npz")
            if os.path.exists(buf_path):
                buffer.load(buf_path)
                print(f"[train] Resumed from checkpoint-{start_step}"
                      f" (buffer: {len(buffer)} states)")
            else:
                print(f"[train] Resumed from checkpoint-{start_step}"
                      f" (buffer file not found, starting empty)")
        else:
            print("[train] No checkpoint found, starting fresh")

    # MCTS config
    mcts_config = MCTSConfig(
        max_simulations=cfg.max_simulations,
        batch_size=cfg.mcts_batch_size,
        uct_c=cfg.uct_c,
        policy_epsilon=cfg.policy_epsilon,
        policy_alpha=cfg.policy_alpha,
        verbose=False,
    )

    _LATEST = -999  # sentinel step for "latest" weights (actors reload this)

    # ── Spawn actors (if multi-process) ──────────────────────────────────
    actors = []
    if cfg.num_actors > 1:
        # Save initial weights so actors can load them
        model.save_checkpoint(_LATEST)
        cfg_dict = asdict(cfg)
        cfg_dict["path"] = cfg.path
        for i in range(cfg.num_actors):
            q = mp.Queue(maxsize=200)
            p = mp.Process(target=actor_process,
                           args=(cfg_dict, q), name=f"actor-{i}")
            p.start()
            actors.append((p, q))
        print(f"[train] Spawned {cfg.num_actors} actor processes")

    # Training state
    global_rng = np.random.RandomState(cfg.seed + start_step)

    try:
        for step in range(start_step + 1, cfg.max_steps + 1):
            t0 = time.time()

            # ── Self-play ──────────────────────────────────────────────
            total_states = 0
            total_games = 0
            outcomes = {"p0": 0, "p1": 0, "draw": 0}

            if cfg.num_actors == 1:
                # Single-process path
                evaluator = PyTorchEvaluator(game, model)
                mcts = BatchMCTS(game, mcts_config, evaluator,
                                 random_state=np.random.RandomState(
                                     cfg.seed + step * 1000))

                while total_states < samples_per_step:
                    record = (total_games == 0)
                    if record:
                        states_info, returns, trace = play_game(
                            game, mcts, cfg, global_rng, record=True)
                    else:
                        states_info, returns = play_game(
                            game, mcts, cfg, global_rng)
                    game_outcome_p0 = returns[0]

                    for obs, mask, policy, cur_player in states_info:
                        buffer.append(obs, mask, policy, game_outcome_p0)

                    if game_outcome_p0 > 0:
                        outcomes["p0"] += 1
                    elif game_outcome_p0 < 0:
                        outcomes["p1"] += 1
                    else:
                        outcomes["draw"] += 1

                    total_states += len(states_info)
                    total_games += 1

                    if record:
                        trace_path = os.path.join(
                            cfg.path, f"game_trace_step{step:04d}.log")
                        with open(trace_path, "w", encoding="utf-8") as f:
                            f.write(f"Step {step} recorded game\n"
                                    f"Temperature: {cfg.temperature}, "
                                    f"temperature_drop:"
                                    f" {cfg.temperature_drop}\n")
                            for line in trace:
                                f.write(line + "\n")
                        print(f"  [trace] wrote {trace_path}")

            else:
                # Multi-actor path: collect from subprocess queues
                while total_states < samples_per_step:
                    for _, q in actors:
                        try:
                            states_info, returns = q.get_nowait()
                        except queue.Empty:
                            continue
                        game_outcome_p0 = returns[0]

                        for obs, mask, policy, cur_player in states_info:
                            buffer.append(obs, mask, policy, game_outcome_p0)

                        if game_outcome_p0 > 0:
                            outcomes["p0"] += 1
                        elif game_outcome_p0 < 0:
                            outcomes["p1"] += 1
                        else:
                            outcomes["draw"] += 1

                        total_states += len(states_info)
                        total_games += 1
                        if total_states >= samples_per_step:
                            break
                    time.sleep(0.001)

            selfplay_time = time.time() - t0

            # ── Training ───────────────────────────────────────────────
            train_t0 = time.time()
            losses_list = []

            for _ in range(n_updates):
                batch = buffer.sample(cfg.train_batch_size)
                loss = model.update(batch)
                losses_list.append(loss)

            if losses_list:
                avg_loss = Losses(
                    policy=sum(l.policy for l in losses_list) / len(losses_list),
                    value=sum(l.value for l in losses_list) / len(losses_list),
                    l2=sum(l.l2 for l in losses_list) / len(losses_list),
                )
            else:
                avg_loss = None
            train_time = time.time() - train_t0

            elapsed = time.time() - t0
            states_per_s = total_states / max(selfplay_time, 0.001)

            # ── Logging ────────────────────────────────────────────────
            log_line = (
                f"[step {step:3d}/{cfg.max_steps}] "
                f"games={total_games:3d}  states={total_states:4d}  "
                f"buffer={len(buffer):5d}/{buffer.total_seen:5d}  "
                f"states/s={states_per_s:.1f}  "
                f"selfplay={selfplay_time:.1f}s  train={train_time:.1f}s"
            )
            if avg_loss is not None:
                log_line += f"  |  loss={avg_loss}"
            log_line += (f"  |  p0_wins={outcomes['p0']}"
                         f"  p1_wins={outcomes['p1']}"
                         f"  draws={outcomes['draw']}")
            print(log_line)

            # ── Checkpoint ─────────────────────────────────────────────
            if step % cfg.checkpoint_freq == 0:
                ckpt_path = model.save_checkpoint(step)
                buffer.save(os.path.join(
                    cfg.path, f"buffer-checkpoint-{step}.npz"))
                print(f"  [checkpoint] Saved {ckpt_path}")

            # Broadcast latest weights to actors
            if cfg.num_actors > 1:
                model.save_checkpoint(_LATEST)

            # ── Evaluation ─────────────────────────────────────────────
            if step % (cfg.checkpoint_freq * 5) == 0 or step == start_step + 1:
                # Run evaluation in background thread so actor queues
                # keep being consumed during the eval games.
                eval_model = build_model(game, cfg)
                eval_model.load_checkpoint(step if step % cfg.checkpoint_freq == 0
                                           else _LATEST)
                eval_future = threading.Thread(
                    target=_run_eval,
                    args=(game, eval_model, cfg, step),
                    daemon=True)
                eval_future.start()

        # Final save
        model.save_checkpoint(cfg.max_steps)
        print(f"\n[train] Done. Final checkpoint: {cfg.max_steps}")

    finally:
        for p, q in actors:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)

    _tee.close()


if __name__ == "__main__":
    main()
