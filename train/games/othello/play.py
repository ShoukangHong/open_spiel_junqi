"""Othello-specific self-play logic: play_game."""

import numpy as np

from train.batch_mcts.mcts import compute_solved_policy
from train.core.weak_move import try_weak_move


def play_game(game, mcts, config, rng, logger=None,
              init_state=None, allow_weak=True):
    """Play one self-play game using BatchMCTS."""
    states_info = []
    rare_games = []
    wstats = {"rare": 0, "weak": 0, "weak_final": 0}
    state = game.new_initial_state() if init_state is None else init_state.clone()
    move_num = 0
    weak_enabled = allow_weak and init_state is None

    # Weak-move setup
    weak_side = None
    weak_count = 0
    weak_max = config.weak_max_per_game if weak_enabled else 0
    weak_steps = set()

    if weak_max > 0 and rng.random() < config.weak_side_prob:
        weak_side = rng.choice([0, 1])
        n_weak = 1
        while n_weak < weak_max and rng.random() < config.weak_move_prob:
            n_weak += 1
        max_step = max(config.temperature_drop + 1, config.weak_move_max_step)
        cand = list(range(config.temperature_drop, max_step))
        if len(cand) >= n_weak:
            weak_steps = set(rng.choice(cand, size=n_weak, replace=False))

    if logger is not None:
        logger.log_game_start(weak_side)
        if weak_steps:
            logger.log_line(f"[weak steps = {sorted(weak_steps)}]")

    while not state.is_terminal():
        cur_player = state.current_player()
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes)
            action = rng.choice(action_list, p=prob_list)
            state.apply_action(action)
            continue

        root = mcts.mcts_search(state)

        # Weak-move
        if ((move_num in weak_steps or (move_num + 1) in weak_steps)
                and weak_side is not None
                and cur_player == weak_side
                and weak_count < weak_max):
            action, tag, weak_count, rare_state, weak_cat = try_weak_move(
                mcts, state, root, config, weak_count, weak_max, logger)
        else:
            action, tag, weak_cat = root.best_child().action, "", ""
            rare_state = None
        if weak_cat in wstats:
            wstats[weak_cat] += 1
        if rare_state is not None:
            rare_games.append(rare_state)

        mcts_action = root.best_child().action

        policy_dict = compute_solved_policy(
            root.children, state.current_player(), game.max_utility())
        policy = np.zeros(game.num_distinct_actions(), dtype=np.float32)
        for a, p in policy_dict.items():
            policy[a] = p
        p_sum = policy.sum()
        if p_sum > 0:
            policy /= p_sum
        else:
            for a in state.legal_actions():
                policy[a] = 1.0
            policy /= policy.sum()

        # Store raw visit-proportional policy
        obs = np.asarray(state.observation_tensor(), dtype=np.float32)
        mask = np.asarray(state.legal_actions_mask(), dtype=bool)
        states_info.append((obs, mask, policy, cur_player, tag))

        # Action selection with temperature
        after_drop = move_num >= config.temperature_drop
        tau_sel = config.temperature if after_drop else 1.0
        if tau_sel > 0 and tau_sel != 1.0:
            sel_probs = policy.astype(np.float64) ** (1.0 / tau_sel)
            sel_probs /= sel_probs.sum()
        else:
            sel_probs = policy

        if tag == "rare" or action != mcts_action:
            pass
        elif after_drop:
            action = mcts_action
        else:
            action = rng.choice(len(sel_probs), p=sel_probs)

        if logger is not None:
            children_sorted = sorted(
                root.children, key=lambda c: c.explore_count, reverse=True)
            top5 = []
            for c in children_sorted[:5]:
                a_str = state.action_to_string(cur_player, c.action)
                top5.append(f"{a_str}(N={c.explore_count} Q={c.q_value:+.3f})")
            mcts_info = " | ".join(top5)
            action_str = state.action_to_string(cur_player, action)
            logger.log_move(move_num + 1, cur_player, state, mcts_info,
                            action_str, tag=tag, tau=tau_sel)

        state.apply_action(action)
        move_num += 1

    returns = state.returns()
    if logger is not None:
        logger.log_game_end(returns, move_num, len(rare_games))
    return states_info, returns, rare_games, wstats


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
