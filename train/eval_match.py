"""Evaluate two strategies against each other in Othello.

Usage: edit the MATCH config below and run.

    python eval_match.py

Strategies:
    "random"   — uniform random over legal actions
    "greedy"   — one-step lookahead: pick the move that maximizes disk count
    "model"    — checkpoint policy head only (no MCTS)
    "mcts"     — checkpoint + BatchMCTS with configurable simulations
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

# ── Strategy config per player ───────────────────────────────────────────────
# Each player can have their own checkpoint, MCTS budget, and strategy.
PLAYER = {
    0: {  # Black (X)
        "strategy":   "mcts",
        "checkpoint_dir":  r"C:\Users\shouk\othello_train_v2",
        "checkpoint_step": 75,
        "mcts_simulations": 128,
        "mcts_batch_size":  4,
        "mcts_uct_c":       1.41,
    },
    1: {  # White (O)
        "strategy":   "mcts",
        "checkpoint_dir":  r"C:\Users\shouk\othello_train_v2",
        "checkpoint_step": 50,
        "mcts_simulations": 128,
        "mcts_batch_size":  4,
        "mcts_uct_c":       1.41,
    },
}

NUM_GAMES = 100
MODEL_TEMPERATURE = 0.03  # for "model" policy sampling (0 = argmax)

# ── Internals ────────────────────────────────────────────────────────────────
_game = None
_models = {}
_mcts_bots = {}
_mcts_evaluators = {}


def _game_obj():
    global _game
    if _game is None:
        _game = pyspiel.load_game("othello")
    return _game


def _load_if_needed(cfg):
    s = cfg["strategy"]
    if s == "model" or s == "mcts":
        _model_for(cfg)



def _model_for(player_cfg):
    """Lazy-load model for a player config dict."""
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
            nn_width=tc["nn_width"], nn_depth=tc["nn_depth"])
        m = Model(net, device=tc.get("device", "cpu"),
                  checkpoint_path=player_cfg["checkpoint_dir"])
        step = player_cfg.get("checkpoint_step", 0)
        if step > 0:
            m.load_checkpoint(step)
            print(f"[{player_cfg['checkpoint_dir'].rsplit(chr(92),1)[-1]}:{step}]"
                  f" params={m.num_trainable_variables}")
        _models[key] = m
    return _models[key]


def _mcts_for(player_cfg):
    """Lazy-load MCTS bot for a player config dict."""
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
            game, cfg, ev, random_state=np.random.RandomState())
    return _mcts_bots[key]


def _act(player_cfg, state):
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
        mask = np.asarray(state.legal_actions_mask(), dtype=np.bool)
        _, policy = _model_for(player_cfg).inference(obs, mask)
        probs = np.array([policy[a] for a in legal])
        if MODEL_TEMPERATURE > 0:
            probs = probs ** (1.0 / MODEL_TEMPERATURE)
            probs /= probs.sum()
            return np.random.choice(legal, p=probs)
        return legal[int(np.argmax(probs))]

    if strategy == "mcts":
        return _mcts_for(player_cfg).step(state)

    raise ValueError(f"Unknown strategy: {strategy}")


def main():
    cfg0, cfg1 = PLAYER[0], PLAYER[1]
    _load_if_needed(cfg0)
    _load_if_needed(cfg1)

    s0, s1 = cfg0["strategy"], cfg1["strategy"]
    name0 = f"{s0}" if s0 in ("random","greedy") else f"{s0}({cfg0.get('mcts_simulations',0)}sim)"
    name1 = f"{s1}" if s1 in ("random","greedy") else f"{s1}({cfg1.get('mcts_simulations',0)}sim)"
    print(f"\nMatch: {name0} (black) vs {name1} (white), {NUM_GAMES} games\n")

    score = {name0: 0, name1: 0, "draw": 0}
    for i in range(NUM_GAMES):
        if i % 2 == 0:
            cfg_b, cfg_w = cfg0, cfg1
            label_b, label_w = name0, name1
        else:
            cfg_b, cfg_w = cfg1, cfg0
            label_b, label_w = name1, name0

        state = _game_obj().new_initial_state()
        while not state.is_terminal():
            cur = state.current_player()
            cfg = cfg_b if cur == 0 else cfg_w
            state.apply_action(_act(cfg, state))
        r = state.returns()[0]

        if r > 0:
            score[label_b] += 1
        elif r < 0:
            score[label_w] += 1
        else:
            score["draw"] += 1

        if (i + 1) % 10 == 0:
            print(f"  {i + 1:4d}/{NUM_GAMES}  "
                  f"{name0}={score[name0]:3d}  {name1}={score[name1]:3d}  "
                  f"draw={score['draw']:3d}")

    n = NUM_GAMES
    print(f"\n-- {name0}: {score[name0]} ({score[name0]/n:.1%})  "
          f"{name1}: {score[name1]} ({score[name1]/n:.1%})  "
          f"draw: {score['draw']} ({score['draw']/n:.1%})")


if __name__ == "__main__":
    main()
