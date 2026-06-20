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
    exp_adv = np.exp((adv[legal] - max_adv) / T)
    softmax = np.zeros(num_actions, dtype=np.float64)
    for i, a in enumerate(legal):
        softmax[a] = exp_adv[i] / exp_adv.sum()

    mixed = (1.0 - alpha) * visit_policy.astype(np.float64) + alpha * softmax
    return (mixed / mixed.sum()).astype(np.float32)


def select_action_with_adv(root, state, temperature=2/3, alpha=0.3, adv_t=0.2,
                           base_policy=None):
    """Select an action using visit-based policy + optional advantage mixing.

    If *base_policy* is given (dict {action: prob}), it is used directly
    instead of deriving from explore_count.  This lets non-MCTS searchers
    (e.g. alpha-beta) supply their own policy.
    """
    n_act = state.get_game().num_distinct_actions()
    probs = np.zeros(n_act, dtype=np.float64)
    if base_policy is not None:
        for a, p in base_policy.items():
            probs[a] = p
    else:
        for c in root.children:
            probs[c.action] = c.explore_count
        s = probs.sum()
        if s <= 0:
            for c in root.children:
                probs[c.action] = c.prior
            s = probs.sum()
        probs /= s
        if alpha > 0 and root.outcome is None:
            probs = mix_advantage(probs, root, state, n_act, alpha, adv_t)
    tau = max(temperature, 0.01)
    probs = probs ** (1.0 / tau)
    probs /= probs.sum()
    legal = [c.action for c in root.children]
    return np.random.choice(legal, p=probs[legal] / probs[legal].sum()), probs
