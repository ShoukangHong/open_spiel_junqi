"""Weak-move exploration — game-agnostic core logic.

Reusable for any pyspiel game.  The caller (play_game) handles
game-specific setup like choosing the weak side and pre-rolling
move steps.
"""


def nn_raw_after_move(evaluator, state, action):
    """Apply *action* to a clone of *state*, then evaluate with NN.

    Returns the NN value as a scalar (via evaluator.scalar_value).
    """
    s = state.clone()
    s.apply_action(action)
    if hasattr(evaluator, 'scalar_value'):
        return evaluator.scalar_value(s)
    # Fallback for duck-typed evaluators (tests)
    nn_val, _ = evaluator._inference(s)
    return float(nn_val) if not hasattr(nn_val, '__len__') else float(nn_val[0])


def try_weak_move(mcts, state, root, config, weak_count, weak_max,
                  logger=None):
    """Attempt a weak move on *state*.

    Caller guarantees: weak_count < weak_max and this is the weak side.

    Returns (action, tag, weak_count, rare_state, weak_cat).
    """
    cur_player = state.current_player()
    mcts_action = root.best_child().action
    action = mcts_action
    tag = ""
    rare_state = None
    weak_cat = ""

    _, nn_policy_arr = mcts.evaluator._inference(state)
    legal = state.legal_actions()
    weak_a = max(legal, key=lambda a: nn_policy_arr[a])

    if weak_a == mcts_action:
        return mcts_action, "", weak_count, None, ""

    nn_val_after = nn_raw_after_move(mcts.evaluator, state, weak_a)
    mcts_val = root.total_reward / max(root.explore_count, 1)
    nn_cur = nn_val_after if cur_player == 0 else -nn_val_after

    def _to_prob(v):
        return max((v + 1.0) / 2.0, 0.005)
    p_before = _to_prob(mcts_val)
    p_after = _to_prob(nn_cur)
    rel_drop = abs(p_before - p_after) / max(p_before, p_after, 0.01)

    if rel_drop > config.rare_case_threshold:
        weak_cat = "rare"
        rare_state = state.clone()
        rare_state.apply_action(weak_a)
        tag = "rare"
        weak_count = weak_max + 1
        if logger:
            logger.log_line(
                f"\n── Rare Case ──\n{state}\n"
                f"  weak action  : {state.action_to_string(cur_player, weak_a)} (a{weak_a})\n"
                f"  MCTS action  : {state.action_to_string(cur_player, mcts_action)} (a{mcts_action})\n"
                f"  nn_cur={nn_cur:.3f}  mcts_val={mcts_val:.3f}  rel_drop={rel_drop:.3f}")
    elif rel_drop < config.weak_move_threshold:
        weak_cat = "weak"
        action = weak_a
        weak_count += 1
    else:
        weak_cat = "weak_final"
        action = weak_a
        weak_count = weak_max + 1

    return action, tag, weak_count, rare_state, weak_cat


# ── Weak-move stats (used by training loop) ──────────────────────────────────

def accum_wstats(cfg, per_game):
    if not hasattr(cfg, "_wstats_total"):
        cfg._wstats_total = {"rare": 0, "weak": 0, "weak_final": 0}
        cfg._wstats_games = 0
    for k in cfg._wstats_total:
        cfg._wstats_total[k] += per_game.get(k, 0)
    cfg._wstats_games += 1


def wstats_summary(cfg):
    if not hasattr(cfg, "_wstats_total") or cfg._wstats_games == 0:
        return "weak=(none)"
    total = cfg._wstats_total
    games = cfg._wstats_games
    return (f"rare={total['rare']:d} wf={total['weak_final']:d}"
            f" weak={total['weak']:d}  ({games}d games)"
            f"  |  rare/g={total['rare']/games:.1f}"
            f" wf/g={total['weak_final']/games:.1f}"
            f" weak/g={total['weak']/games:.1f}")


def reset_wstats(cfg):
    if hasattr(cfg, "_wstats_total"):
        del cfg._wstats_total
        del cfg._wstats_games
