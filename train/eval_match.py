"""Evaluate two strategies against each other in Othello.

Usage: edit the MATCH config below and run.

    python eval_match.py

Strategies:
    "random"   — uniform random over legal actions
    "greedy"   — one-step lookahead: pick the move that maximizes disk count
    "model"    — checkpoint policy head only (no MCTS)
    "mcts"     — checkpoint + BatchMCTS with configurable simulations
"""

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

# ── Match config ──────────────────────────────────────────────────────────
CHECKPOINT_DIR = r"C:\Users\shouk\othello_train_v2"
CHECKPOINT_STEP = 25        # 0 = random model (skip loading)
MODEL_WIDTH = 32
MODEL_DEPTH = 8
NUM_GAMES = 100

BLACK = "model"    # strategy for black (player 0)
WHITE = "random"  # strategy for white (player 1)

# ── Strategy parameters ────────────────────────────────────────────────────
MODEL_TEMPERATURE = 0.1    # temperature for "model" policy sampling (0 = argmax)
MCTS_SIMULATIONS = 64      # playouts for "mcts" strategy
MCTS_BATCH_SIZE = 4       # batch size for MCTS leaf evaluation
MCTS_UCT_C = 1.41
MCTS_VERBOSE = False

# ── Internals ───────────────────────────────────────────────────────────────
_game = None
_model = None
_mcts_evaluator = None
_mcts_bot = None


def _game_obj():
    global _game
    if _game is None:
        _game = pyspiel.load_game("othello")
    return _game


def _model_obj():
    global _model
    if _model is None:
        game = _game_obj()
        obs_shape = game.observation_tensor_shape()
        num_actions = game.num_distinct_actions()

        net = OthelloResNet(
            input_channels=obs_shape[0],
            board_size=obs_shape[1],
            output_size=num_actions,
            nn_width=MODEL_WIDTH,
            nn_depth=MODEL_DEPTH,
        )
        m = Model(net, checkpoint_path=CHECKPOINT_DIR)
        if CHECKPOINT_STEP > 0:
            m.load_checkpoint(CHECKPOINT_STEP)
            print(f"Loaded checkpoint-{CHECKPOINT_STEP}  "
                  f"params={m.num_trainable_variables}")
        else:
            print(f"Using random model  params={m.num_trainable_variables}")
        _model = m
    return _model


def _mcts():
    global _mcts_evaluator, _mcts_bot
    if _mcts_bot is None:
        game = _game_obj()
        _mcts_evaluator = PyTorchEvaluator(game, _model_obj())
        mcts_cfg = MCTSConfig(
            max_simulations=MCTS_SIMULATIONS,
            batch_size=MCTS_BATCH_SIZE,
            uct_c=MCTS_UCT_C,
            policy_epsilon=0,   # no noise during evaluation
            verbose=MCTS_VERBOSE,
        )
        _mcts_bot = BatchMCTS(game, mcts_cfg, _mcts_evaluator,
                              random_state=np.random.RandomState())
    return _mcts_bot


def _act(strategy, state):
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
        _, policy = _model_obj().inference(obs, mask)
        probs = np.array([policy[a] for a in legal])
        if MODEL_TEMPERATURE > 0:
            probs = probs ** (1.0 / MODEL_TEMPERATURE)
            probs /= probs.sum()
            return np.random.choice(legal, p=probs)
        return legal[int(np.argmax(probs))]

    if strategy == "mcts":
        return _mcts().step(state)

    raise ValueError(f"Unknown strategy: {strategy}")


def main():
    if "model" in (BLACK, WHITE) or "mcts" in (BLACK, WHITE):
        _model_obj()

    print(f"Match: {BLACK} (black) vs {WHITE} (white), {NUM_GAMES} games\n")

    score = {BLACK: 0, WHITE: 0, "draw": 0}
    for i in range(NUM_GAMES):
        if i % 2 == 0:
            strat_b, strat_w = BLACK, WHITE
        else:
            strat_b, strat_w = WHITE, BLACK

        state = _game_obj().new_initial_state()
        while not state.is_terminal():
            cur = state.current_player()
            strategy = strat_b if cur == 0 else strat_w
            state.apply_action(_act(strategy, state))
        r = state.returns()[0]

        if r > 0:
            score[strat_b] += 1
        elif r < 0:
            score[strat_w] += 1
        else:
            score["draw"] += 1

        if (i + 1) % 10 == 0:
            print(f"  {i + 1:4d}/{NUM_GAMES}  "
                  f"{BLACK}={score[BLACK]:3d}  {WHITE}={score[WHITE]:3d}  "
                  f"draw={score['draw']:3d}")

    n = NUM_GAMES
    print(f"\n-- {BLACK}: {score[BLACK]} ({score[BLACK]/n:.1%})  "
          f"{WHITE}: {score[WHITE]} ({score[WHITE]/n:.1%})  "
          f"draw: {score['draw']} ({score['draw']/n:.1%})")


if __name__ == "__main__":
    main()
