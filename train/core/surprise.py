"""Surprise detection — uses KL divergence to find MCTS vs NN mismatch.

Reads cached NN values from node.nn_q / node.nn_draw / children.prior,
so no extra NN inference is needed.
"""

import numpy as np
from train.batch_mcts.mcts import compute_solved_policy


def _wdl_from_qdr(q, draw_rate):
    """Reconstruct WDL distribution from Q and draw rate."""
    w = max((q + 1.0 - draw_rate) / 2.0, 0.0)
    l = max((1.0 - q - draw_rate) / 2.0, 0.0)
    s = w + draw_rate + l
    if s > 0:
        return np.array([w / s, draw_rate / s, l / s], dtype=np.float64)
    return np.array([0.0, 1.0, 0.0], dtype=np.float64)


def _kl(p, q):
    """KL divergence KL(p||q) with epsilon smoothing."""
    eps = 1e-12
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def detect_surprise(state, root, config, game_max_utility=1.0):
    """Return (root_tag, child_tags) based on MCTS vs NN KL divergence.

    Policy surprise: KL(MCTS_pol || NN_prior) > surprise_pol_kl.
    Value surprise:  KL(MCTS_WDL || NN_WDL) > surprise_val_kl.

    root_tag: "" | "surprise" | "strong_surprise"
    child_tags: {action: "child_surprise"}
    """
    if root.nn_q is None or not root.children:
        return "", {}, 0.0

    cur = state.current_player()
    mcts_q = root.q_value
    nn_q = root.nn_q

    # ── Policy KL ─────────────────────────────────────────────────────
    actions = [c.action for c in root.children]
    nn_prior = np.array([max(c.prior, 0.0) for c in root.children],
                        dtype=np.float64)
    nn_prior /= nn_prior.sum()
    solved = compute_solved_policy(
        root.children, cur, game_max_utility,
        root_visits=root.explore_count)
    mcts_pol = np.array([solved.get(a, 0.0) for a in actions],
                        dtype=np.float64)
    mcts_pol /= mcts_pol.sum()
    # KL(search || prior): how much MCTS diverges from NN expectation
    pol_kl = _kl(mcts_pol, nn_prior)

    # ── Value KL ──────────────────────────────────────────────────────
    nn_draw = root.nn_draw if root.nn_draw is not None else 0.0
    nn_wdl = _wdl_from_qdr(nn_q, nn_draw)
    mcts_draw = root.draw_rate
    mcts_wdl = _wdl_from_qdr(mcts_q, mcts_draw)
    val_kl = _kl(mcts_wdl, nn_wdl)

    # ── Root tag ──────────────────────────────────────────────────────
    combined = pol_kl + val_kl
    if combined > config.surprise_pol_kl + config.surprise_val_kl:
        root_tag = "super_surprise"
    elif pol_kl > config.surprise_pol_kl or val_kl > config.surprise_val_kl:
        root_tag = "surprise"
    else:
        root_tag = ""

    # ── Child tags ─────────────────────────────────────────────────────
    child_tags = {}
    min_n = config.surprise_child_min_n
    for c in root.children:
        if c.nn_q is None:
            continue
        if c.outcome is not None or c.explore_count >= min_n:
            c_nn_draw = c.nn_draw if c.nn_draw is not None else 0.0
            c_nn_wdl = _wdl_from_qdr(c.nn_q, c_nn_draw)
            c_mcts_draw = c.draw_rate
            c_mcts_wdl = _wdl_from_qdr(c.q_value, c_mcts_draw)
            if _kl(c_mcts_wdl, c_nn_wdl) > config.surprise_val_kl:
                child_tags[c.action] = "child_surprise"

    return root_tag, child_tags, combined
