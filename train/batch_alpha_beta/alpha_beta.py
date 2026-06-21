"""Batch Alpha-Beta search with neural-network move ordering and evaluation.

Children are sorted by NN prior, then evaluated in mini-batches.
Alpha-beta pruning stops further batches once the bound is reached.
"""

import math
import time

import numpy as np
import pyspiel

from train.batch_alpha_beta.node import AlphaBetaNode
from train.core.position_hash import hash_state


class BatchAlphaBeta:
    """Alpha-Beta search with lazy mini-batch NN evaluation per node."""

    def __init__(self, game, evaluator=None, depth=4, batch_size=8,
                 random_state=None, policy_temp=0.2):
        game_type = game.get_type()
        if game_type.reward_model != pyspiel.GameType.RewardModel.TERMINAL:
            raise ValueError("Game must have terminal rewards.")
        if game_type.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
            raise ValueError("Game must have sequential turns.")
        self._game = game
        self._evaluator = evaluator
        self._depth = depth
        self._batch_size = batch_size
        self._policy_temp = policy_temp
        self.max_utility = game.max_utility()
        self._random_state = random_state or np.random.RandomState()

    # ── Public API ──────────────────────────────────────────────────────

    def search(self, state, root=None):
        _ = root
        self._stats = dict(nn_calls=0, nn_states=0, nn_time=0.0,
                           nodes=0, depth_max=0, prunes=0)
        t0 = time.time()
        root_player = state.current_player()
        root_node, _ = self._alpha_beta(
            state, 0, -float("inf"), float("inf"), root_player)
        ab_time = time.time() - t0
        root_node.action = None
        root_node.player = root_player
        root_node.explore_count = sum(
            1 for c in root_node.children if c.explore_count > 0)
        s = self._stats
        print(f"[AB] depth={self._depth}  batch={self._batch_size}  "
              f"nodes={s['nodes']}  max_depth={s['depth_max']}  "
              f"prunes={s['prunes']}  "
              f"nn_calls={s['nn_calls']}  nn_states={s['nn_states']}  "
              f"nn_time={s['nn_time']:.2f}s  ab_time={ab_time:.2f}s  "
              f"total={s['nn_time']+ab_time:.2f}s", flush=True)
        return root_node

    def compute_root_policy(self, root, state=None):
        return self.compute_value_policy(root, self._policy_temp)

    def mcts_search(self, state, root=None):
        return self.search(state, root=root)

    def step(self, state):
        return self.step_with_policy(state)[1]

    @staticmethod
    def compute_value_policy(root, policy_temp=0.2):
        if not root.children:
            return {}
        V = root.q_value
        acts = [c.action for c in root.children]
        adv = np.array([c.q_value - V for c in root.children], dtype=np.float64)
        adv -= adv.max()
        probs = np.exp(adv / policy_temp)
        probs /= probs.sum()
        return {a: float(p) for a, p in zip(acts, probs)}

    def step_with_policy(self, state, temperature=0.0):
        if state.is_chance_node():
            return [(pyspiel.INVALID_ACTION, 1.0)], pyspiel.INVALID_ACTION
        root = self.search(state)
        best = root.best_child()
        if best is None:
            legal = state.legal_actions()
            n = len(legal)
            return [(a, 1.0 / n) for a in legal], legal[0]
        p_dict = self.compute_value_policy(root, self._policy_temp)
        policy = [(a, p_dict.get(a, 0.0)) for a in sorted(p_dict.keys())]
        if temperature > 0 and len(policy) > 1:
            actions, probs = zip(*policy)
            probs = np.array(probs, dtype=np.float64)
            probs = probs ** (1.0 / temperature)
            probs /= probs.sum()
            action = actions[self._random_state.choice(len(actions), p=probs)]
        else:
            action = best.action
        return policy, action

    # ── Internal ────────────────────────────────────────────────────────

    @staticmethod
    def _scalar(wdl):
        return float(wdl[0] - wdl[2])

    def _nn_eval(self, states):
        """Batch-evaluate *states* via NN.  Returns (wdl_arr, priors_list)."""
        if not states or self._evaluator is None:
            n = len(states)
            vals = np.full((n, 3), [0.0, 1.0, 0.0], dtype=np.float32)
            priors = [[] for _ in range(n)]
            return vals, priors
        t0 = time.time()
        vals, priors = self._evaluator.batch_inference_raw(states)
        self._stats["nn_calls"] += 1
        self._stats["nn_states"] += len(states)
        self._stats["nn_time"] += time.time() - t0
        return vals, priors

    def _alpha_beta(self, state, depth, alpha, beta, player,
                    node_priors=None, node_wdl=None):
        """Recursive alpha-beta with lazy mini-batch child evaluation."""
        self._stats["nodes"] += 1
        if depth > self._stats["depth_max"]:
            self._stats["depth_max"] = depth
        node = AlphaBetaNode(None, state.current_player(), 1.0)
        cur = state.current_player()
        is_max = (cur == player)

        if state.is_terminal():
            ret = state.returns()
            node.outcome = np.array(ret)
            node.total_reward = ret[player]
            node.draw_reward = 1.0 if all(r == 0 for r in ret) else 0.0
            node.explore_count = 1
            return node, ret[player]

        legal = state.legal_actions()
        if not legal:
            node.total_reward = 0.0
            node.explore_count = 1
            return node, 0.0

        # ── Depth limit ────────────────────────────────────────────────
        if depth >= self._depth:
            if node_wdl is not None:
                scalar = self._scalar(node_wdl)
                if cur != player:
                    scalar = -scalar
                node.total_reward = scalar
            else:
                node.total_reward = 0.0
            node.explore_count = 1
            _pd = dict(node_priors or [])
            for a in legal:
                child = AlphaBetaNode(a, cur, _pd.get(a, 0.0))
                node.children.append(child)
            return node, node.total_reward

        # ── Get priors, sort, chunk into mini-batches ──────────────────
        if node_priors is not None:
            my_prior_dict = dict(node_priors)
        else:
            # Root node: evaluate to get priors
            _, root_priors = self._nn_eval([state])
            my_prior_dict = dict(root_priors[0])

        # Sort actions by prior descending
        scored_actions = sorted(
            legal, key=lambda a: my_prior_dict.get(a, 0.0), reverse=True)
        batches = [scored_actions[i:i + self._batch_size]
                   for i in range(0, len(scored_actions), self._batch_size)]

        # ── Lazy-eval batches, recurse, prune ──────────────────────────
        if is_max:
            best_val = -float("inf")
            for batch_actions in batches:
                # Prepare child states for this batch
                batch_states, batch_term = [], []
                for a in batch_actions:
                    cs = state.clone(); cs.apply_action(a)
                    batch_states.append(cs)
                    batch_term.append(cs.is_terminal())

                # NN-eval non-terminal children
                nonterm = [(j, batch_states[j]) for j in range(len(batch_states))
                           if not batch_term[j]]
                nt_states = [s for _, s in nonterm]
                if nt_states:
                    vals, priors = self._nn_eval(nt_states)

                # Alpha-beta on this batch (in prior order)
                batch_done = False
                for j, a in enumerate(batch_actions):
                    if batch_term[j]:
                        ret = batch_states[j].returns()
                        child = AlphaBetaNode(a, cur, my_prior_dict.get(a, 0.0))
                        child.outcome = np.array(ret)
                        child.total_reward = ret[player]
                        child.draw_reward = (1.0 if all(r == 0 for r in ret)
                                             else 0.0)
                        child.explore_count = 1
                        child_val = ret[player]
                        child_p = None
                    else:
                        # Find this child's result in the NN output
                        nt_idx = next(k for k, (jj, _) in enumerate(nonterm)
                                      if jj == j)
                        wdl = vals[nt_idx]
                        child_p = priors[nt_idx]
                        child, child_val = self._alpha_beta(
                            batch_states[j], depth + 1, alpha, beta, player,
                            node_priors=child_p, node_wdl=wdl)
                    child.action = a
                    child.prior = my_prior_dict.get(a, 0.0)
                    child.player = cur
                    node.children.append(child)
                    if child_val > best_val:
                        best_val = child_val
                    alpha = max(alpha, best_val)
                    if alpha >= beta:
                        self._stats["prunes"] += 1
                        batch_done = True
                        break
                if batch_done:
                    break  # don't evaluate remaining batches
        else:  # MIN
            best_val = float("inf")
            for batch_actions in batches:
                batch_states, batch_term = [], []
                for a in batch_actions:
                    cs = state.clone(); cs.apply_action(a)
                    batch_states.append(cs)
                    batch_term.append(cs.is_terminal())

                nonterm = [(j, batch_states[j]) for j in range(len(batch_states))
                           if not batch_term[j]]
                nt_states = [s for _, s in nonterm]
                if nt_states:
                    vals, priors = self._nn_eval(nt_states)

                batch_done = False
                for j, a in enumerate(batch_actions):
                    if batch_term[j]:
                        ret = batch_states[j].returns()
                        child = AlphaBetaNode(a, cur, my_prior_dict.get(a, 0.0))
                        child.outcome = np.array(ret)
                        child.total_reward = ret[player]
                        child.draw_reward = (1.0 if all(r == 0 for r in ret)
                                             else 0.0)
                        child.explore_count = 1
                        child_val = ret[player]
                        child_p = None
                    else:
                        nt_idx = next(k for k, (jj, _) in enumerate(nonterm)
                                      if jj == j)
                        wdl = vals[nt_idx]
                        child_p = priors[nt_idx]
                        child, child_val = self._alpha_beta(
                            batch_states[j], depth + 1, alpha, beta, player,
                            node_priors=child_p, node_wdl=wdl)
                    child.action = a
                    child.prior = my_prior_dict.get(a, 0.0)
                    child.player = cur
                    node.children.append(child)
                    if child_val < best_val:
                        best_val = child_val
                    beta = min(beta, best_val)
                    if alpha >= beta:
                        self._stats["prunes"] += 1
                        batch_done = True
                        break
                if batch_done:
                    break

        node.total_reward = best_val
        node.explore_count = 1
        return node, best_val
