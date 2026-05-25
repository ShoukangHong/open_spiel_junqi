"""Batch MCTS Evaluators.

Provides:
  - BatchEvaluator (abstract) — interface for batch NN evaluation
  - RandomRolloutEvaluator — baseline (mirrors original MCTS version)
  - PyTorchEvaluator — wraps PyTorch model for batch MCTS

The original AlphaZeroEvaluator (JAX) from OpenSpiel is kept for reference.
"""

from typing import Any

import numpy as np

from open_spiel.python.algorithms import mcts as orig_mcts
import pyspiel
from open_spiel.python.utils import lru_cache

# Re-export original classes so existing tests still work.
Evaluator = orig_mcts.Evaluator
RandomRolloutEvaluator = orig_mcts.RandomRolloutEvaluator
SearchNode = orig_mcts.SearchNode
MCTSBot = orig_mcts.MCTSBot


# ── Batch Evaluator interface ──────────────────────────────────────────────

class BatchEvaluator:
    """Abstract evaluator that supports batched inference.

    The batch methods are the primary interface for BatchMCTS.
    Single-sample methods delegate to batch for convenience.
    """

    def batch_evaluate(self, states: list) -> np.ndarray:
        """Returns values for a list of states.

        Args:
            states: list of pyspiel.State.

        Returns:
            ndarray of shape (len(states), num_players) or (len(states),).
        """
        raise NotImplementedError

    def batch_prior(self, states: list) -> list:
        """Returns priors for a list of states.

        Args:
            states: list of pyspiel.State.

        Returns:
            list of [(action, prob), ...] — one per state.
        """
        raise NotImplementedError


# ── Random rollout evaluator (batch-aware, for testing) ────────────────────

class BatchRandomRolloutEvaluator(BatchEvaluator):
    """Random rollout evaluator exposing the BatchEvaluator interface.

    No NN involved — useful for correctness testing of BatchMCTS.
    """

    def __init__(self, n_rollouts: int = 1,
                 random_state: np.random.RandomState = None,
                 max_length: int = None):
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
        """Batch evaluate + prior.

        Returns:
            values: (len(states),) scalar values from each state's
                    current player's perspective.
            priors: list of [(action, prob), ...].
        """
        full_returns = self.batch_evaluate(states)  # (batch, 2)
        # Pick the return for the player about to move at each leaf.
        scalar_values = np.array([
            float(full_returns[i, state.current_player()])
            for i, state in enumerate(states)
        ])
        return scalar_values, self.batch_prior(states)

    # Single-sample interface (for compat with OriginalMCTS)
    def evaluate(self, state):
        return self._eval_one(state)

    def prior(self, state):
        return self.batch_prior([state])[0]


# ── PyTorch model evaluator ────────────────────────────────────────────────

class PyTorchEvaluator(BatchEvaluator):
    """Wraps a PyTorch Model for batch MCTS evaluation."""

    def __init__(self, game: pyspiel.Game, model: Any,
                 cache_size: int = 2**16):
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
        """Single-state inference with LRU cache."""
        key = self._make_cache_key(state)
        value, policy = self._cache.make(
            key,
            lambda: self._model.inference(
                np.asarray(state.observation_tensor(), dtype=np.float32),
                np.asarray(state.legal_actions_mask(), dtype=bool)),
        )
        return value, policy

    def evaluate(self, state):
        """Single-state value."""
        value, _ = self._inference(state)
        return np.array([value, -value])

    def prior(self, state):
        if state.is_chance_node():
            return state.chance_outcomes()
        _, policy = self._inference(state)
        return [(a, float(policy[a])) for a in state.legal_actions()]

    def batch_evaluate(self, states):
        """Batch value evaluation (cached single-inference per state)."""
        values = []
        for state in states:
            value, _ = self._inference(state)
            values.append([value, -value])
        return np.array(values)

    def batch_prior(self, states):
        """Batch prior evaluation."""
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
        """Direct batch inference bypassing the LRU cache.

        Used inside BatchMCTS for fresh leaf nodes.
        Returns (values_array, list_of_policy_lists).
        """
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

        # Convert policies array → list of [(action, prob), ...]
        prior_list = []
        for state, policy_arr in zip(states, policies):
            prior_list.append(
                [(a, float(policy_arr[a])) for a in state.legal_actions()])
        return values, prior_list


# ── Original JAX AlphaZeroEvaluator (kept for reference) ───────────────────

class AlphaZeroEvaluator(orig_mcts.Evaluator):
    """Original JAX AlphaZero MCTS Evaluator — unchanged from OpenSpiel."""

    def __init__(self, game: pyspiel.Game, model: Any,
                 cache_size: int = 2**16) -> None:
        if game.num_players() != 2:
            raise ValueError("Game must be for two players.")
        game_type = game.get_type()
        if game_type.reward_model != pyspiel.GameType.RewardModel.TERMINAL:
            raise ValueError("Game must have terminal rewards.")
        if game_type.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
            raise ValueError("Game must have sequential turns.")

        self._model = model
        self._cache = lru_cache.LRUCache(cache_size)

    def cache_info(self):
        return self._cache.info()

    def clear_cache(self):
        self._cache.clear()

    def _inference(self, state):
        obs = np.asarray(state.observation_tensor())
        mask = np.asarray(state.legal_actions_mask())
        cache_key = obs.tobytes() + mask.tobytes()
        value, policy = self._cache.make(
            cache_key, lambda: self._model.inference(obs, mask))
        return value, policy

    def evaluate(self, state):
        value, _ = self._inference(state)
        return np.array([value, -value])

    def prior(self, state):
        if state.is_chance_node():
            return state.chance_outcomes()
        _, policy = self._inference(state)
        return [(action, policy[action]) for action in state.legal_actions()]
