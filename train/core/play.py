"""Shared self-play logic — game-agnostic.

Provides play_game() which drives one episode of self-play using BatchMCTS,
including weak-move exploration, rare-case forking, and early pruning.
"""

import numpy as np

from train.batch_mcts.mcts import compute_solved_policy
from train.core.policy import mix_advantage
from train.core.weak_move import try_weak_move
from train.core.surprise import detect_surprise

_EPS = 1e-6  # label smoothing — prevents float32 underflow from CE with p=0


def _smooth_policy(policy, legal):
    """Blend uniform noise so no action has zero probability."""
    n = len(legal)
    if n == 0:
        return policy
    u = _EPS / n
    policy = policy * (1.0 - _EPS)
    policy[legal] += u
    return policy


def assign_players(rng, mcts_main, mcts_best, mcts_opp=None,
                   best_model_prob=0.3, random_opponent_prob=0.2,
                   use_best=None, use_opp=None):
    """Pick models for p0 and p1 in one self-play game.

    p0 is always main.  p1 is best (prob best_model_prob), random_opp
    (prob random_opponent_prob, best not selected), or main otherwise.
    If p1 is not main, colours are swapped with 50% chance.

    When *use_best* / *use_opp* are given (replaying a rare fork), those
    values are used directly instead of rolling new random draws.

    Returns:
        (mcts_p0, mcts_p1, use_best, use_opp)
    """
    if use_best is None:
        use_best = (best_model_prob > 0 and rng.random() < best_model_prob)
    if use_opp is None:
        use_opp = (not use_best and mcts_opp is not None
                   and rng.random() < random_opponent_prob)

    p0, p1 = mcts_main, mcts_main
    if use_best:
        p1 = mcts_best
    elif use_opp:
        p1 = mcts_opp

    if (use_best or use_opp) and rng.random() < 0.5:
        p0, p1 = p1, p0

    return p0, p1, use_best, use_opp


def _stable_qdr(root):
    """Q and draw_rate from children with >1 visit, excluding forced-explore noise.

    Root children that were visited only once (e.g. uniform-expand coverage)
    add noise to the value target.  Filtering them out gives a stabler V.
    """
    valid = [c for c in root.children if c.explore_count > 1]
    if valid:
        total_n = sum(c.explore_count for c in valid)
        q = sum(c.total_reward for c in valid) / total_n
        dr = sum(c.draw_reward for c in valid) / total_n
    else:
        q = root.q_value
        dr = root.draw_rate
    return q, dr


def _should_prune(pruned, root, cur_player, config, max_utility, dice):
    """Decide whether to prune (win) or truncate (draw) the current game.

    Returns (new_pruned, do_break).
    Checks every move; dice is rolled each time.
    """
    if not config.prune_enabled or pruned["used"]:
        return pruned, False

    proven_win = root.outcome is not None and root.outcome[cur_player] > 0
    high_q = root.q_value >= config.prune_threshold * max_utility
    draw_rate = getattr(root, 'draw_rate', 0.0) or 0.0
    high_draw = draw_rate >= config.prune_threshold

    qualifies = proven_win or high_q or high_draw
    if not qualifies:
        return pruned, False

    if dice < config.prune_prob:
        is_draw = high_draw and not (proven_win or high_q)
        return {"used": True, "cur_player": cur_player,
                "draw_truncate": is_draw}, True
    return pruned, False


def _setup_weak_moves(config, rng, *, allow_weak=True):
    """Setup weak-move parameters for one game.

    Returns (weak_side, weak_max, weak_steps).
    """
    weak_side = None
    weak_max = config.weak_max_per_game if allow_weak else 0
    weak_steps = set()

    if weak_max > 0 and rng.random() < config.weak_side_prob:
        weak_side = rng.choice([0, 1])
        n_weak = 1
        while n_weak < weak_max and rng.random() < config.weak_move_prob:
            n_weak += 1
        max_step = config.weak_move_max_step
        cand = list(range(2, max_step))
        if len(cand) >= n_weak:
            weak_steps = set(rng.choice(cand, size=n_weak, replace=False))
    return weak_side, weak_max, weak_steps


def play_game(game, mcts_black, mcts_white, config, rng, logger=None,
              init_state=None, allow_weak=True):
    """Play one self-play game using BatchMCTS.

    mcts_black / mcts_white may be the same object (standard self-play)
    or different (cross-play vs historical best).

    Returns:
        (states_info, returns, rare_games, wstats)
    """
    states_info = []
    extra_surprise = []  # collected here, appended at end (keeps game sequence clean)
    rare_games = []
    wstats = {"rare": 0, "weak": 0, "weak_final": 0, "rare_flip": 0}
    state = game.new_initial_state() if init_state is None else init_state.clone()
    move_num = 0
    weak_enabled = allow_weak and init_state is None

    weak_side, weak_max, weak_steps = _setup_weak_moves(
        config, rng, allow_weak=weak_enabled)
    weak_count = 0

    if logger is not None:
        logger.log_game_start(weak_side)
        if weak_steps:
            logger.log_line(f"[weak steps = {sorted(weak_steps)}]")

    pruned = {"used": False, "cur_player": None}

    while not state.is_terminal():
        cur_player = state.current_player()
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes)
            action = rng.choice(action_list, p=prob_list)
            state.apply_action(action)
            continue

        mcts = mcts_black if cur_player == 0 else mcts_white
        root = mcts.mcts_search(state)

        pruned, do_break = _should_prune(
            pruned, root, cur_player, config, game.max_utility(),
            rng.random())
        if do_break:
            if logger is not None:
                logger.log_line(
                    f"[prune] step {move_num} p{cur_player} "
                    f"Q={root.q_value:+.3f} — early win\n{state}")
            # Record pruned state before exiting
            obs = np.asarray(state.observation_tensor(), dtype=np.float32)
            mask = np.asarray(state.legal_actions_mask(), dtype=bool)
            policy_dict = compute_solved_policy(
                root.children, cur_player, game.max_utility(),
                root_visits=root.explore_count)
            policy = np.zeros(game.num_distinct_actions(), dtype=np.float32)
            for a, p in policy_dict.items():
                policy[a] = p
            policy_sum = policy.sum()
            if policy_sum > 0:
                policy /= policy_sum
            else:
                policy[...] = 1.0 / policy.size
            policy = _smooth_policy(policy, state.legal_actions())
            if root.outcome is not None:
                q = root.outcome[cur_player]
                dr = 1.0 if q == 0 else 0.0
            else:
                q, dr = _stable_qdr(root)
            states_info.append((obs, mask, policy, cur_player, "", q, dr))
            break

        # Weak-move
        if ((move_num in weak_steps or (move_num + 1) in weak_steps)
                and weak_side is not None
                and cur_player == weak_side
                and weak_count < weak_max):
            action, tag, weak_count, rare_state, weak_cat = try_weak_move(
                mcts, state, root, config, weak_count, weak_max, logger, rng)
        else:
            action, tag, weak_cat = root.best_child().action, "", ""
            rare_state = None

        if weak_cat in wstats:
            wstats[weak_cat] += 1
        if rare_state is not None:
            rare_games.append(rare_state)

        mcts_action = root.best_child().action

        policy_dict = compute_solved_policy(
            root.children, state.current_player(), game.max_utility(),
            root_visits=root.explore_count)
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
        policy = _smooth_policy(policy, state.legal_actions())

        # Advantage mixing for unsolved states
        if (config.policy_mix_alpha > 0
                and root.outcome is None
                and root.explore_count > 0):
            policy = mix_advantage(policy, root, state,
                                   game.num_distinct_actions(),
                                   config.policy_mix_alpha,
                                   config.adv_temperature)

        obs = np.asarray(state.observation_tensor(), dtype=np.float32)
        mask = np.asarray(state.legal_actions_mask(), dtype=bool)
        if root.outcome is not None:
            q = root.outcome[cur_player]
            dr = 1.0 if q == 0 else 0.0
        else:
            q, dr = _stable_qdr(root)
        states_info.append((obs, mask, policy, cur_player, tag, q, dr))

        # Surprise detection: collect candidates (resolved at end of game)
        if (getattr(config, 'surprise_val_kl', 0) > 0
                and root.nn_q is not None and not tag):
            surprise_tag, child_tags, kl_sum = detect_surprise(
                state, root, config, game.max_utility())
            if surprise_tag:
                extra_surprise.append(
                    (obs.copy(), mask.copy(), policy.copy(),
                     cur_player, surprise_tag, q, dr, kl_sum))

        # Action selection with temperature
        # Forked (rare) games: always use post-drop tau for clean evaluation
        after_drop = (init_state is not None
                      or move_num >= config.temperature_drop)
        tau_sel = config.temperature if after_drop else 0.5
        if tau_sel > 0.01 and tau_sel != 1.0:
            sel_probs = policy.astype(np.float64) ** (1.0 / tau_sel)
            sel_probs /= sel_probs.sum()
        elif 0 < tau_sel <= 0.01:
            sel_probs = np.zeros(len(policy), dtype=np.float64)
            sel_probs[policy.argmax()] = 1.0
        else:
            sel_probs = policy

        if tag == "rare" or action != mcts_action:
            pass  # keep the weak/rare action
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

    if pruned.get("draw_truncate"):
        returns = np.array([0.0, 0.0], dtype=np.float64)
    elif pruned["used"]:
        max_u = game.max_utility()
        p = pruned["cur_player"]
        returns = np.array([max_u if i == p else -max_u for i in range(2)],
                           dtype=np.float64)
    else:
        returns = state.returns()
    if logger is not None:
        logger.log_game_end(returns, move_num, len(rare_games))
    # Generate extra copies: top-3 super_surprise ×3, rest ×1, surprise ×1
    extra_surprise.sort(key=lambda x: x[7], reverse=True)
    super_count = 0
    copies = []
    for item in extra_surprise:
        tag = item[4]
        if tag == "super_surprise":
            super_count += 1
            n = 3 if super_count <= 3 else 1
            final_tag = "super_surprise" if super_count <= 3 else "surprise"
        else:
            n = 1
            final_tag = tag
        for _ in range(n):
            copies.append((item[0], item[1], item[2], item[3],
                           final_tag, item[5], item[6], item[7]))
    max_copies = max(10, len(states_info) // 5)
    if len(copies) > max_copies:
        copies = copies[:max_copies]
    states_info.extend(copies)
    return states_info, returns, rare_games, wstats
