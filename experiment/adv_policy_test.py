"""Test advantage-mixed policy output for various Q differences."""
import sys, os
import numpy as np
import pyspiel

from train.batch_mcts.node import Node
from train.core.policy import select_action_with_adv

ALPHA = 0.3
ADV_T = 0.2
TAU = 2/3
SAMPLES = 5000

game = pyspiel.load_game("tic_tac_toe")
state = game.new_initial_state()

for q_a, q_b, label in [
    (0.9, 0.95, "diff=-0.05"),
    (0.9, 0.85, "diff=0.05"),
    (0.9, 0.8,  "diff=0.1"),
    (0.9, 0.4,  "diff=0.5"),
    (0.9, -0.2, "diff=1.1"),
    (0.5, -0.5, "diff=1.0 (0.5 vs -0.5)"),
]:
    root = Node(None, state.current_player(), 1)
    root.children = []
    default_q = 0.0     # Q for unexplored actions
    default_n = 1       # 0 visits
    total_other = 0     # total reward contribution from unexplored
    root.explore_count = 100
    root.total_reward = 92 * q_a + 1 * q_b + 7 * default_q
    for a in state.legal_actions():
        if a == 0:
            q, n = q_a, 99
        elif a == 1:
            q, n = q_b, 1
        else:
            q, n = default_q, default_n
        c = Node(a, root.player, prior=0.5)
        c.explore_count = n
        c.total_reward = q * n
        root.children.append(c)
        total_other += q * n

    counts = {}
    for _ in range(SAMPLES):
        a, _ = select_action_with_adv(root, state, alpha=ALPHA, adv_t=ADV_T,
                                   temperature=TAU)
        counts[a] = counts.get(a, 0) + 1

    p_a = counts.get(0, 0) / SAMPLES
    p_b = counts.get(1, 0) / SAMPLES
    p_other = sum(v for k, v in counts.items() if k not in (0, 1)) / SAMPLES
    print(f"alpha={ALPHA}  tau={TAU}  adv_t={ADV_T}")
    print(f"  Q:  A={q_a:+.3f}  B={q_b:+.3f}  (others Q={default_q})")
    print(f"  policy:  A={p_a:.4f}  B={p_b:.4f}  others={p_other:.4f}")
    print()
