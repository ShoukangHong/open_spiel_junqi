"""Debug drawable flag: replay position and check if p1 has a drawable move."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pyspiel
from train.batch_mcts.config import MCTSConfig
from train.batch_mcts.mcts import BatchMCTS
from train.core.model_builder import build_xiangqi_model
from train.core.position_hash import hash_state

# Load position from saved sequence
with open(r"C:\Users\shouk\xiangqi_train\cloud_sov\saved_positions\stick.txt") as f:
    moves = [int(line.strip()) for line in f if line.strip()]

game = pyspiel.load_game("xiangqi")
state = game.new_initial_state()
for a in moves:
    state.apply_action(a)

cur = state.current_player()
print(f"Turn: {'RED(p0)' if cur == 0 else 'BLACK(p1)'}")
print(f"Move#: {len(state.history())}")
print(f"Legal actions: {len(state.legal_actions(cur))}")
print(f"History (last 10): {[state.action_to_string(cur, a) for a in state.history()[-10:]]}")
print(f"\nPosition:\n{state}")

# Build model + MCTS (same as game_ui config)
cfg_dict = {"nn_width": 128, "nn_depth": 10, "device": "cpu",
            "learning_rate": 1e-3, "weight_decay": 1e-4,
            "game": "xiangqi",
            "path": r"C:\Users\shouk\xiangqi_train\cloud_sov"}
m = build_xiangqi_model(game, cfg_dict)
m.load_checkpoint(325)

mcts_cfg = MCTSConfig(
    max_simulations=2000, batch_size=10, uct_c=6.0,
    policy_epsilon=0.0, policy_alpha=0.25,
    draw_penalty=0.05, repeat_penalty=0.2,
    fpu_lambda=0.2, probe_depth=1, probe_surprise=0.3,
    verbose=False,
    solve=True,
)
from train.batch_mcts.evaluator import PyTorchEvaluator
ev = PyTorchEvaluator(game, m)
mcts = BatchMCTS(game, mcts_cfg, ev)

# Instrument backprop to verify drawable clamp
_orig_bp = mcts._backprop
_log = []
def _patched_bp(path, returns, draw_prob=0.0):
    clamped_happened = False
    clamped_orig = False
    for i in range(len(path) - 1, -1, -1):
        node = path[i]
        if node.drawable and returns[node.player] < 0:
            clamped_happened = True
            if not clamped_orig:
                clamped_orig = True
    if clamped_happened:
        _log.append(1)
    return _orig_bp(path, returns, draw_prob)
mcts._backprop = _patched_bp

root = mcts.mcts_search(state)

print(f"\nBackprop clamp count: {sum(_log)}/{len(_log)} paths")
mcts._backprop = _orig_bp

print(f"\nRoot: outcome={root.outcome} drawable={root.drawable}")
print(f"  explore_count={root.explore_count}  Q={root.q_value:+.4f}  dr={root.draw_rate:.3f}")
print(f"\nChildren (top N by visits):")
for c in sorted(root.children, key=lambda c: c.explore_count, reverse=True)[:8]:
    act_str = state.action_to_string(cur, c.action)
    print(f"  {act_str:>10s}  N={c.explore_count:>5d}  Q={c.q_value:+.4f}"
          f"  dr={c.draw_rate:.3f}  outcome={c.outcome}  drawable={c.drawable}")
    if c.children:
        # check grandchild for draw
        any_draw = any(cc.outcome is not None and all(r == 0 for r in cc.outcome) for cc in c.children)
        any_drawable = any(cc.drawable for cc in c.children)
        print(f"         grandchild_draw={any_draw}  grandchild_drawable={any_drawable}"
              f"  grandchild_outcomes={[cc.outcome.tolist() if cc.outcome is not None else None for cc in c.children[:3]]}")

# Detail: child (6,7)-(3,7)
target = next(c for c in root.children
              if state.action_to_string(cur, c.action) == "R(6,7)-(3,7)")
if target is not None and target.children:
    print(f"\n--- Grandchildren of {state.action_to_string(cur, target.action)} ---")
    for gc in sorted(target.children, key=lambda c: c.explore_count, reverse=True):
        gc_str = target.state.action_to_string(
            target.state.current_player(), gc.action) if target.state else str(gc.action)
        print(f"  {gc_str:>12s}  N={gc.explore_count:>5d}  Q={gc.q_value:+.4f}"
              f"  dr={gc.draw_rate:.3f}  outcome={gc.outcome}  drawable={gc.drawable}")

# Check: R's total_reward and draw_reward vs explore_count
target_R = next(c for c in root.children
                if state.action_to_string(cur, c.action) == "R(6,7)-(3,7)")
print(f"\nR(6,7)-(3,7) details:")
print(f"  total_reward={target_R.total_reward:.2f}")
print(f"  draw_reward={target_R.draw_reward:.2f}")
print(f"  explore_count={target_R.explore_count}")
print(f"  q_value={target_R.q_value:.4f}")
print(f"  draw_rate={target_R.draw_rate:.4f}")
print(f"  drawable={target_R.drawable}")
print(f"  outcome={target_R.outcome}")
if target_R.drawable:
    # If drawable were correctly clamping, total_reward ≈ 0
    print(f"  Expected Q≈0 (drawable clamp), actual Q={target_R.q_value:+.4f}")

# Also check h(3,1)-(5,2) total_reward
target_h = next(c for c in target_R.children
                if target_R.state.action_to_string(
                    target_R.state.current_player(),
                    c.action) == "h(3,1)-(5,2)")
print(f"\nh(3,1)-(5,2) details:")
print(f"  total_reward={target_h.total_reward:.2f}")
print(f"  draw_reward={target_h.draw_reward:.2f}")
print(f"  explore_count={target_h.explore_count}")
print(f"  q_value={target_h.q_value:.4f}")
print(f"  drawable={target_h.drawable}")

print("complete")