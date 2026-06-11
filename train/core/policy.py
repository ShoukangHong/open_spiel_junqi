"""Policy target computation utilities."""
import math

import numpy as np


def mix_advantage(visit_policy, root, state, num_actions, alpha,
                  temperature=0.2):
    """Advantage mixing: pi'(a) ∝ (1-alpha)·pi_mcts(a) + alpha·softmax(A/T).

    A(s,a) = Q(s,a) - V(s) where V(s) = root.q_value.
    Lower temperature amplifies small advantage differences.
    Only valid for unsolved states (root.outcome is None).
    """
    V = root.q_value
    adv = np.zeros(num_actions, dtype=np.float64)
    max_adv = -1e9
    for c in root.children:
        A = c.q_value - V
        adv[c.action] = A
        if A > max_adv:
            max_adv = A
    legal = state.legal_actions()
    T = max(temperature, 0.01)
    exp_adv = np.exp((adv[legal] - max_adv) / T)  # exp_adv = np.exp((adv[legal] - max_adv) / (T * math.sqrt(1 - abs(V)))) 不确定是否有用
    softmax = np.zeros(num_actions, dtype=np.float64)
    for i, a in enumerate(legal):
        softmax[a] = exp_adv[i] / exp_adv.sum()

    mixed = (1.0 - alpha) * visit_policy.astype(np.float64) + alpha * softmax
    return (mixed / mixed.sum()).astype(np.float32)
