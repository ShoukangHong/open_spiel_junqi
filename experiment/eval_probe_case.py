"""Quick probe evaluation on a specific position.

Usage: python -m experiment.eval_probe_case
"""
import numpy as np
import pyspiel

from game_ui.core import load_model_for_ui, create_bot, print_search_info
from game_ui.xiangqi_render import action_label

CHECKPOINT_DIR = r"C:\Users\shouk\xiangqi_train\cloud_b"
CHECKPOINT_STEP = 65
MCTS_SIMS = 1024
BATCH_SIZE = 8
UCT_C = 3.0
FPU_LAMBDA = 0.2
PROBE_DEPTH = 3
PROBE_SURPRISE = 1.0

from train.core.model_builder import build_xiangqi_model
model, game = load_model_for_ui(CHECKPOINT_DIR, CHECKPOINT_STEP, build_xiangqi_model)

# Replay history
state = game.new_initial_state()
case_path = r"C:\Users\shouk\xiangqi_train\cases\paochouju.txt"
with open(case_path) as f:
    for line in f:
        a = int(line.strip())
        if a in state.legal_actions():
            state.apply_action(a)
        else:
            print(f"Skip illegal action {a}")
            break

print(f"Position after {len(state.history())} moves, "
      f"player={'Red' if state.current_player()==0 else 'Black'}, "
      f"legal={len(state.legal_actions())}")

# Probe bot
bot, evaluator, _ = create_bot(
    game, model, MCTS_SIMS, BATCH_SIZE, UCT_C,
    fpu_lambda=FPU_LAMBDA,
    probe_depth=PROBE_DEPTH, probe_surprise=PROBE_SURPRISE)
root = bot.mcts_search(state.clone())
policy = bot.compute_root_policy(root, state)

print_search_info(root, state, policy,
                  evaluator=evaluator, action_label_fn=action_label,
                  max_moves=30, engine_label="Probe")

# Find target: (3,7)→(3,4)
target_label = "(3,7)→(3,4)"
for c in root.children:
    if action_label(c.action) == target_label:
        print(f"\n  >>> Target: {target_label}")
        print(f"  N={c.explore_count}  Q={c.q_value:+.4f}  prior={c.prior:.4f}")
        if c.state and not c.state.is_terminal():
            vals, _ = evaluator.batch_inference_raw([c.state])
            q_nn = float(vals[0][0] - vals[0][2])
            p = c.state.current_player()
            if p != state.current_player():
                q_nn = -q_nn
            print(f"  Vnn(root): w={vals[0][0]:.3f} d={vals[0][1]:.3f} l={vals[0][2]:.3f}"
                  f"  q={q_nn:+.4f}")
        break
