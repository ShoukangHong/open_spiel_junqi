"""Evaluate two strategies against each other in Othello.

Usage:  python eval_match.py

Also importable:  from train.eval_match import run_match
"""

import json
import sys
import os
_sys_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _sys_root not in sys.path:
    sys.path.insert(0, _sys_root)

import numpy as np
import pyspiel

from train.batch_mcts.config import MCTSConfig
from train.batch_mcts.evaluator import PyTorchEvaluator
from train.batch_mcts.mcts import BatchMCTS
from train.model.othello_resnet import Model, OthelloResNet

# ── Module-level caches (reused across run_match calls) ──────────────────────
_game = None
_models = {}
_mcts_bots = {}
_mcts_evaluators = {}


def _game_obj():
    global _game
    if _game is None:
        _game = pyspiel.load_game("othello")
    return _game


def _model_for(player_cfg):
    key = (player_cfg["checkpoint_dir"], player_cfg.get("checkpoint_step", 0))
    if key not in _models:
        game = _game_obj()
        config_path = os.path.join(player_cfg["checkpoint_dir"], "train_config.json")
        with open(config_path) as f:
            tc = json.load(f)
        net = OthelloResNet(
            input_channels=game.observation_tensor_shape()[0],
            board_size=game.observation_tensor_shape()[1],
            output_size=game.num_distinct_actions(),
            nn_width=tc.get("nn_width", 32),
            nn_depth=tc.get("nn_depth", 6))
        m = Model(net, device=tc.get("device", "cpu"),
                  checkpoint_path=player_cfg["checkpoint_dir"])
        step = player_cfg.get("checkpoint_step", 0)
        if step > 0:
            m.load_checkpoint(step)
            print(f"  [eval] loaded {player_cfg['checkpoint_dir'].rsplit(chr(92),1)[-1].rsplit('/',1)[-1]}:{step}"
                  f"  params={m.num_trainable_variables}")
        _models[key] = m
    return _models[key]


def _mcts_for(player_cfg):
    key = id(player_cfg)
    if key not in _mcts_bots:
        game = _game_obj()
        ev = PyTorchEvaluator(game, _model_for(player_cfg))
        _mcts_evaluators[key] = ev
        cfg = MCTSConfig(
            max_simulations=player_cfg.get("mcts_simulations", 128),
            batch_size=player_cfg.get("mcts_batch_size", 4),
            uct_c=player_cfg.get("mcts_uct_c", 1.41),
            policy_epsilon=0, verbose=False)
        _mcts_bots[key] = BatchMCTS(
            game, cfg, ev, random_state=np.random.RandomState(42))
    return _mcts_bots[key]


def _act(player_cfg, state, move_num, temperature, temp_drop):
    strategy = player_cfg["strategy"]
    legal = state.legal_actions()

    if strategy == "random":
        return np.random.choice(legal)

    if strategy == "greedy":
        cur = state.current_player()
        token = "x" if cur == 0 else "o"
        counts = [str(state.clone().apply_action(a)).count(token)
                  for a in legal]
        return legal[int(np.argmax(counts))]

    if strategy == "model":
        obs = np.asarray(state.observation_tensor(), dtype=np.float32)
        mask = np.asarray(state.legal_actions_mask(), dtype=bool)
        _, policy = _model_for(player_cfg).inference(obs, mask)
        probs = np.array([policy[a] for a in legal])
        if temperature > 0:
            probs = probs ** (1.0 / temperature)
            probs /= probs.sum()
            return np.random.choice(legal, p=probs)
        return legal[int(np.argmax(probs))]

    if strategy == "mcts":
        root = _mcts_for(player_cfg).mcts_search(state)
        visits = np.array([c.explore_count for c in root.children])
        probs = visits / visits.sum()
        tau = 1.0 if move_num < temp_drop else temperature
        probs = probs ** (1.0 / max(tau, 0.01))
        probs /= probs.sum()
        actions = [c.action for c in root.children]
        return np.random.choice(actions, p=probs)

    raise ValueError(f"Unknown strategy: {strategy}")


# ── Core function ────────────────────────────────────────────────────────────

def run_match(cfg0, cfg1, num_games=100, temperature=0.1, temp_drop=4,
              quiet=True):
    """Run a match between two strategies.

    Args:
        cfg0, cfg1: dicts with keys: strategy, checkpoint_dir, checkpoint_step,
                    mcts_simulations, mcts_batch_size, mcts_uct_c.
        num_games: number of games to play.
        temperature: model/MCTS temperature after temp_drop.
        temp_drop: first N moves use τ=1 for MCTS.
        quiet: if True, suppress per-player load messages.

    Returns:
        (score_dict, sequences)
        score_dict: {"name0": wins, "name1": wins, "draw": draws}
        sequences: list of (label, tuple_of_actions)
    """
    s0, s1 = cfg0["strategy"], cfg1["strategy"]
    st0 = cfg0.get("checkpoint_step", 0)
    st1 = cfg1.get("checkpoint_step", 0)
    name0 = (f"{s0}" if s0 in ("random", "greedy")
             else f"{s0}(step{st0},{cfg0.get('mcts_simulations', 0)}sim)")
    name1 = (f"{s1}" if s1 in ("random", "greedy")
             else f"{s1}(step{st1},{cfg1.get('mcts_simulations', 0)}sim)")

    # Pre-load models (unconditionally to populate caches)
    for cfg in (cfg0, cfg1):
        s = cfg["strategy"]
        if s in ("model", "mcts"):
            _model_for(cfg)

    if not quiet:
        print(f"\nMatch: {name0} (black) vs {name1} (white), {num_games} games\n")

    score = {name0: 0, name1: 0, "draw": 0}
    sequences = []

    for i in range(num_games):
        if i % 2 == 0:
            cfg_b, cfg_w = cfg0, cfg1
            label_b, label_w = name0, name1
        else:
            cfg_b, cfg_w = cfg1, cfg0
            label_b, label_w = name1, name0

        state = _game_obj().new_initial_state()
        move_num = 0
        moves = []
        while not state.is_terminal():
            cur = state.current_player()
            cfg = cfg_b if cur == 0 else cfg_w
            action = _act(cfg, state, move_num, temperature, temp_drop)
            state.apply_action(action)
            moves.append(action)
            move_num += 1
        sequences.append((label_b, tuple(moves)))
        r = state.returns()[0]

        if r > 0:
            score[label_b] += 1
        elif r < 0:
            score[label_w] += 1
        else:
            score["draw"] += 1

        if not quiet and (i + 1) % 10 == 0:
            print(f"  {i + 1:4d}/{num_games}  "
                  f"{name0}={score[name0]:3d}  {name1}={score[name1]:3d}  "
                  f"draw={score['draw']:3d}")

    return score, sequences


# ── Default config + main ────────────────────────────────────────────────────

DEFAULT_NUM_GAMES = 100
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TEMP_DROP = 7


def main():
    score, sequences = run_match(
        PLAYER[0], PLAYER[1],
        num_games=DEFAULT_NUM_GAMES,
        temperature=DEFAULT_TEMPERATURE,
        temp_drop=DEFAULT_TEMP_DROP,
        quiet=False)

    n = DEFAULT_NUM_GAMES
    s0, s1 = list(score.keys())[:2]
    print(f"\n-- {s0}: {score[s0]} ({score[s0]/n:.1%})  "
          f"{s1}: {score[s1]} ({score[s1]/n:.1%})  "
          f"draw: {score['draw']} ({score['draw']/n:.1%})")

    # Move diversity
    print("\n[PREFIX OVERLAP]  unique prefixes / games  (most-common count)")
    max_len = max(len(m) for _, m in sequences)
    for L in range(1, min(max_len + 1, 61), 5):
        prefs = [m[:L] for _, m in sequences if len(m) >= L]
        if not prefs:
            continue
        arr = np.array(prefs, dtype=int)
        unique, counts = np.unique(arr, axis=0, return_counts=True)
        top_count = counts.max()
        pct = top_count / len(prefs) * 100
        print(f"  first {L:>2d} moves:  {len(unique)} unique  "
              f"(top occurs {top_count}/{len(prefs)} = {pct:.0f}%)")


PLAYER = {
    # 0: {"strategy": "mcts",
    #     "checkpoint_dir": r"C:\Users\shouk\othello_train_v2",
    #     "checkpoint_step": 130,
    #     "mcts_simulations": 128, "mcts_batch_size": 6, "mcts_uct_c": 1.41},
    0: {"strategy": "mcts",
        "checkpoint_dir": r"C:\Users\shouk\othello_train_cloud",
        "checkpoint_step": 180,
        "mcts_simulations": 128, "mcts_batch_size": 8, "mcts_uct_c": 1.41},
    1: {"strategy": "mcts",
        "checkpoint_dir": r"C:\Users\shouk\othello_train_cloud",
        "checkpoint_step": 180,
        "mcts_simulations": 512, "mcts_batch_size": 32, "mcts_uct_c": 1.41},
}

if __name__ == "__main__":
    main()
