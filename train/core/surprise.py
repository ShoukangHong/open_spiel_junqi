"""Surprise detection — uses KL divergence to find MCTS vs NN mismatch.

Reads cached NN values from node.nn_q / node.nn_draw / children.prior,
so no extra NN inference is needed.
"""

import numpy as np
from train.batch_mcts.mcts import compute_solved_policy


def _stable_qdr(node):
    """Q and draw_rate from well-explored children only (≥10% of max visits).

    Falls back to node.q_value / node.draw_rate when no child qualifies.
    """
    if node.outcome is not None:
        s = node.state
        p = s.current_player() if s is not None and not s.is_terminal() else node.player
        q = node.outcome[p]
        return q, 1.0 if q == 0 else 0.0
    if not node.children:
        return node.q_value, node.draw_rate
    max_n = max(c.explore_count for c in node.children)
    if max_n == 0:
        return node.q_value, node.draw_rate
    threshold = max_n * 0.1
    if max_n > 1:
        threshold = max(threshold, 1.0)
    valid = [c for c in node.children if c.explore_count > threshold]
    if valid:
        total_n = sum(c.explore_count for c in valid)
        if total_n > 0:
            q = sum(c.total_reward for c in valid) / total_n
            dr = sum(c.draw_reward for c in valid) / total_n
            return q, dr
    return node.q_value, node.draw_rate


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


def _node_val_kl(node, nn_q, nn_draw):
    """Value KL for a single node (works with or without children)."""
    mcts_q, mcts_draw = _stable_qdr(node)
    _draw = nn_draw if nn_draw is not None else 0.0
    nn_wdl = _wdl_from_qdr(nn_q, _draw)
    mcts_wdl = _wdl_from_qdr(mcts_q, mcts_draw)
    return _kl(mcts_wdl, nn_wdl)


def _node_pol_val_kl(node, player, nn_q, nn_draw, nn_prior_arr,
                     game_max_utility):
    """(pol_kl, val_kl) for a node that has children.

    nn_prior_arr: normalized array aligned with node.children order.
    Applies outcome-based pol_kl adjustments:
      - proven loss for *player* → pol_kl = 0
      - proven win/draw → scale by avg visits per branch
    """
    actions = [c.action for c in node.children]
    solved = compute_solved_policy(
        node.children, player, game_max_utility,
        root_visits=node.explore_count)
    mcts_pol = np.array([solved.get(a, 0.0) for a in actions],
                        dtype=np.float64)
    mcts_pol /= mcts_pol.sum()
    pol_kl = _kl(mcts_pol, nn_prior_arr)

    if node.outcome is not None:
        if node.outcome[player] < 0:
            pol_kl = 0.0
        else:
            n = len(node.children)
            if n > 0:
                avg = node.explore_count / n
                scale = 0.2 + 0.8 * min(avg / 5.0, 1.0)
                pol_kl *= scale

    val_kl = _node_val_kl(node, nn_q, nn_draw)
    return pol_kl, val_kl


def detect_surprise(state, root, config, game_max_utility=1.0):
    """Return (root_tag, child_tags) based on MCTS vs NN KL divergence.

    Policy surprise: KL(MCTS_pol || NN_prior) > surprise_pol_kl.
    Value surprise:  KL(MCTS_WDL || NN_WDL) > surprise_val_kl.

    root_tag: "" | "surprise" | "super_surprise"
    child_tags: {action: ("child_surprise"|"child_super_surprise", combined_kl)}
    """
    if root.nn_q is None or not root.children:
        return "", {}, 0.0

    cur = state.current_player()

    def _surprise_tag(p_kl, v_kl, prefix=""):
        combined = p_kl + v_kl
        if combined > config.surprise_pol_kl + config.surprise_val_kl:
            return prefix + "super_surprise", combined
        if p_kl > config.surprise_pol_kl or v_kl > config.surprise_val_kl:
            return prefix + "surprise", combined
        return "", combined

    # ── Root ──────────────────────────────────────────────────────────
    root_nn_prior = np.array([max(c.prior, 0.0) for c in root.children],
                             dtype=np.float64)
    root_nn_prior /= root_nn_prior.sum()
    pol_kl, val_kl = _node_pol_val_kl(
        root, cur, root.nn_q, root.nn_draw, root_nn_prior, game_max_utility)

    root_tag, combined = _surprise_tag(pol_kl, val_kl)

    # ── Children ──────────────────────────────────────────────────────
    child_tags = {}
    min_n = config.surprise_child_min_n
    for c in root.children:
        if c.nn_q is None or c.nn_prior is None:
            continue
        if c.outcome is not None or c.explore_count >= min_n:
            c_player = (c.state.current_player()
                        if c.state is not None else c.player)
            if c.children:
                c_nn_prior = np.array(
                    [max(dict(c.nn_prior).get(cc.action, 0.0), 0.0)
                     for cc in c.children],
                    dtype=np.float64)
                c_nn_prior /= c_nn_prior.sum()
                c_pol_kl, c_val_kl = _node_pol_val_kl(
                    c, c_player, c.nn_q, c.nn_draw, c_nn_prior,
                    game_max_utility)
            else:
                c_pol_kl = 0.0
                c_val_kl = _node_val_kl(c, c.nn_q, c.nn_draw)
            ctag, c_combined = _surprise_tag(c_pol_kl, c_val_kl,
                                             prefix="child_")
            if ctag:
                child_tags[c.action] = (ctag, c_combined)

    return root_tag, child_tags, combined
