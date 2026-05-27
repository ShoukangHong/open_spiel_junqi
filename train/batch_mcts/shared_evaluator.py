"""Shared GPU evaluator for multi-actor training.

Instead of N actors each doing their own tiny GPU calls (serialised by
CUDA), all actors submit leaf states to a single inference thread that
batches them together for one forward pass.  Designed to support
multi-model mixing in the future via model_id.

Usage:
    server = InferenceServer()
    server.register_model("main", model)
    server.start()

    eval = SharedEvaluator(game, server, "main")
    # use eval in BatchMCTS like PyTorchEvaluator
"""

import queue
import threading

import numpy as np


class _Future:
    """A one-shot future for inference results."""

    __slots__ = ("_event", "_value", "_policy")

    def __init__(self):
        self._event = threading.Event()
        self._value = None
        self._policy = None

    def set(self, value, policy):
        self._value = value
        self._policy = policy
        self._event.set()

    def get(self):
        self._event.wait()
        return self._value, self._policy


class InferenceServer:
    """Background thread that batches leaf-eval requests from all actors.

    Actors submit (states, model_id, future) triples via submit().
    The server thread coalesces submissions into batches grouped by
    model_id, runs a single forward pass per group, and writes results
    back through the futures.
    """

    def __init__(self, batch_timeout: float = 0.002):
        self._batch_timeout = batch_timeout
        self._lock = threading.Lock()
        self._models = {}       # model_id → model (OthelloResNet)
        self._pending = []      # [(states_list, model_id, future)]
        self._thread = None
        self._running = False

    def register_model(self, model_id: str, model):
        with self._lock:
            self._models[model_id] = model

    def submit(self, states, model_id: str, future: _Future):
        with self._lock:
            self._pending.append((states, model_id, future))

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _run(self):
        while self._running:
            # Collect pending submissions
            with self._lock:
                if not self._pending:
                    # Nothing pending — brief sleep then retry
                    pass
                batch = self._pending
                self._pending = []

            if not batch:
                self._event.wait(self._batch_timeout)
                continue

            # Group by model_id
            groups = {}
            for states_list, mid, fut in batch:
                groups.setdefault(mid, []).append((states_list, fut))

            # One forward pass per model group
            for mid, entries in groups.items():
                model = self._models.get(mid)
                if model is None:
                    for _, fut in entries:
                        fut.set(0.0, np.zeros(65, dtype=np.float32))
                    continue

                # Coalesce all states into one batch
                all_states = []
                state_counts = []
                for states_list, _ in entries:
                    all_states.extend(states_list)
                    state_counts.append(len(states_list))

                if not all_states:
                    continue

                obs_batch = np.stack(
                    [np.asarray(s.observation_tensor(), dtype=np.float32)
                     for s in all_states], axis=0)
                mask_batch = np.stack(
                    [np.asarray(s.legal_actions_mask(), dtype=bool)
                     for s in all_states], axis=0)

                values, policies = model.batch_inference(obs_batch, mask_batch)

                # Scatter results back to individual futures
                cursor = 0
                for i, (states_list, fut) in enumerate(entries):
                    n = state_counts[i]
                    for j, state in enumerate(states_list):
                        idx = cursor + j
                        prior = [(a, float(policies[idx, a]))
                                 for a in state.legal_actions()]
                        fut.set(float(values[idx]), prior)
                    cursor += n


class SharedEvaluator:
    """BatchMCTS-compatible evaluator backed by a shared InferenceServer.

    Implements the same interface as PyTorchEvaluator:
      - _inference(state) → (value, policy_array)
      - batch_inference_raw(states) → (values_array, prior_list)

    The batch_inference_raw path is the one BatchMCTS actually calls;
    it submits all leaf states together and blocks until the server
    returns results.
    """

    def __init__(self, game, server: InferenceServer, model_id: str = "main"):
        self._game = game
        self._server = server
        self._model_id = model_id

    def _inference(self, state):
        """Single-state inference (used by _nn_raw_after_move etc.)."""
        fut = _Future()
        self._server.submit([state], self._model_id, fut)
        value, prior = fut.get()
        policy = np.zeros(self._game.num_distinct_actions(), dtype=np.float32)
        for a, p in prior:
            policy[a] = p
        return value, policy

    def batch_inference_raw(self, states):
        """Batch inference — the primary path used by BatchMCTS.

        Submits all leaf states in one go and blocks for results.
        """
        if not states:
            return np.array([]), []

        fut = _Future()
        self._server.submit(states, self._model_id, fut)
        value, prior_list = fut.get()

        return np.array([value], dtype=np.float32), prior_list
