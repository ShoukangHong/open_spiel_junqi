"""Batch MCTS Evaluators.

Provides:
  - BatchEvaluator (abstract) — interface for batch NN evaluation
  - RandomRolloutEvaluator — baseline (mirrors original MCTS version)
  - PyTorchEvaluator — wraps PyTorch model for batch MCTS
"""

from typing import Any

import numpy as np

from open_spiel.python.algorithms import mcts as orig_mcts
import pyspiel
from open_spiel.python.utils import lru_cache

# Re-export original classes
Evaluator = orig_mcts.Evaluator
RandomRolloutEvaluator = orig_mcts.RandomRolloutEvaluator
SearchNode = orig_mcts.SearchNode
MCTSBot = orig_mcts.MCTSBot


# ── Batch Evaluator interface ──────────────────────────────────────────────

class BatchEvaluator:
    """Abstract evaluator that supports batched inference."""

    def scalar_value(self, state) -> float:
        """Return a scalar Q in [-1, 1] regardless of value format.

        Subclasses may override for efficiency; default uses _inference.
        """
        value, _ = self._inference(state)
        if hasattr(value, '__len__'):
            v = np.asarray(value, dtype=np.float32).ravel()
            return float(v[0] - v[2])
        return float(value)

    def batch_evaluate(self, states):
        raise NotImplementedError

    def batch_prior(self, states):
        raise NotImplementedError


# ── Random rollout evaluator (batch-aware, for testing) ────────────────────

class BatchRandomRolloutEvaluator(BatchEvaluator):

    def __init__(self, n_rollouts=1, random_state=None, max_length=None):
        self.n_rollouts = n_rollouts
        self.max_length = max_length
        self._random_state = random_state or np.random.RandomState()

    def _eval_one(self, state):
        result = None
        for _ in range(self.n_rollouts):
            ws = state.clone()
            length = 0
            while not ws.is_terminal():
                if ws.is_chance_node():
                    outcomes = ws.chance_outcomes()
                    action_list, prob_list = zip(*outcomes)
                    action = self._random_state.choice(action_list, p=prob_list)
                else:
                    action = self._random_state.choice(ws.legal_actions())
                ws.apply_action(action)
                length += 1
                if self.max_length is not None and length >= self.max_length:
                    break
            returns = np.array(ws.returns())
            result = returns if result is None else result + returns
        return result / self.n_rollouts

    def batch_evaluate(self, states):
        return np.stack([self._eval_one(s) for s in states])

    def batch_prior(self, states):
        results = []
        for state in states:
            if state.is_chance_node():
                results.append(state.chance_outcomes())
            else:
                legal = state.legal_actions(state.current_player())
                results.append([(a, 1.0 / len(legal)) for a in legal])
        return results

    def batch_inference_raw(self, states):
        full_returns = self.batch_evaluate(states)
        wdl_values = []
        for i, state in enumerate(states):
            v = float(full_returns[i, state.current_player()])
            w = max(v, 0.0)
            l = max(-v, 0.0)
            d = 1.0 - abs(v)
            wdl_values.append([w, d, l])
        return np.array(wdl_values, dtype=np.float32), self.batch_prior(states)

    def evaluate(self, state):
        return self._eval_one(state)

    def prior(self, state):
        return self.batch_prior([state])[0]


# ── PyTorch model evaluator ────────────────────────────────────────────────

class PyTorchEvaluator(BatchEvaluator):

    def __init__(self, game, model, cache_size=2**16):
        self._game = game
        self._model = model
        self._cache = lru_cache.LRUCache(cache_size)
        self._output_size = game.num_distinct_actions()

    def cache_info(self):
        return self._cache.info()

    def clear_cache(self):
        self._cache.clear()

    def _make_cache_key(self, state):
        obs = np.asarray(state.observation_tensor(), dtype=np.float32)
        mask = np.asarray(state.legal_actions_mask(), dtype=bool)
        return obs.tobytes() + mask.tobytes()

    def _inference(self, state):
        key = self._make_cache_key(state)
        value, policy = self._cache.make(
            key,
            lambda: self._model.inference(
                np.asarray(state.observation_tensor(), dtype=np.float32),
                np.asarray(state.legal_actions_mask(), dtype=bool)),
        )
        return value, policy

    def scalar_value(self, state) -> float:
        """Return p0-perspective scalar Q from WDL output."""
        value, _ = self._inference(state)
        v = np.asarray(value, dtype=np.float32).ravel()
        q = float(v[0] - v[2])               # w - l from current player view
        if state.current_player() == 1:
            q = -q                           # flip to p0 view
        return q

    def evaluate(self, state):
        value, _ = self._inference(state)
        return np.array(value)               # [w, d, l]

    def prior(self, state):
        if state.is_chance_node():
            return state.chance_outcomes()
        _, policy = self._inference(state)
        return [(a, float(policy[a])) for a in state.legal_actions()]

    def batch_evaluate(self, states):
        values = []
        for state in states:
            value, _ = self._inference(state)
            values.append([value, -value])
        return np.array(values)

    def batch_prior(self, states):
        results = []
        for state in states:
            if state.is_chance_node():
                results.append(state.chance_outcomes())
            else:
                _, policy = self._inference(state)
                results.append(
                    [(a, float(policy[a])) for a in state.legal_actions()])
        return results

    def batch_inference_raw(self, states):
        if not states:
            return np.array([]), []

        obs_list = []
        mask_list = []
        for state in states:
            obs_list.append(
                np.asarray(state.observation_tensor(), dtype=np.float32))
            mask_list.append(
                np.asarray(state.legal_actions_mask(), dtype=bool))

        obs_batch = np.stack(obs_list, axis=0)
        mask_batch = np.stack(mask_list, axis=0)

        values, policies = self._model.batch_inference(obs_batch, mask_batch)

        prior_list = []
        for state, policy_arr in zip(states, policies):
            legal = state.legal_actions()
            probs = policy_arr[legal]
            prior_list.append(
                list(zip(legal, probs.astype(float).tolist())))
        return values, prior_list
