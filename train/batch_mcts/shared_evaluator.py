"""Shared GPU evaluator for multi-actor training.

Supports multiple models via model_id routing — each actor can use a
different checkpoint without spawning separate GPU processes.

Architecture:
  actor → incoming_q: (actor_id, model_id, obs_list, mask_list)
  Server: groups by model_id, batched forward per group, scatters
  actor ← result_qs[actor_id]: (actor_id, results)
"""

import multiprocessing as mp
import queue
import time

import numpy as np


class SharedEvaluator:
    """BatchMCTS-compatible evaluator backed by a shared GPU process."""

    def __init__(self, game, incoming_q: mp.Queue, result_q: mp.Queue,
                 actor_id: int = 0, model_id: str = "main"):
        self._game = game
        self._incoming = incoming_q
        self._result = result_q
        self._actor_id = actor_id
        self._model_id = model_id

    def scalar_value(self, state):
        """Return p0-perspective scalar Q from WDL output."""
        value, _ = self._inference(state)
        q = float(value[0] - value[2])         # w - l from current player view
        if state.current_player() == 1:
            q = -q                              # flip to p0 view
        return q

    def _inference(self, state):
        val, prior = self._infer_one(state)
        policy = np.zeros(self._game.num_distinct_actions(), dtype=np.float32)
        for a, p in prior:
            policy[a] = p
        return val, policy

    def _infer_one(self, state):
        obs = np.asarray(state.observation_tensor(), dtype=np.float32)
        mask = np.asarray(state.legal_actions_mask(), dtype=bool)
        obs_b = obs.reshape(1, -1)
        mask_b = mask.reshape(1, -1)
        self._incoming.put((self._actor_id, self._model_id, obs_b, mask_b))
        while True:
            aid, mid, data = self._result.get()
            if aid == self._actor_id and mid == self._model_id:
                if isinstance(data, Exception):
                    raise data
                break
        values, priors = self._process_batch(data, mask_b)
        return values[0], priors[0]

    def batch_inference_raw(self, states):
        if not states:
            return np.array([]), []

        obs_list = [np.asarray(s.observation_tensor(), dtype=np.float32)
                    for s in states]
        mask_list = [np.asarray(s.legal_actions_mask(), dtype=bool)
                     for s in states]
        obs_b = np.stack(obs_list, axis=0)
        mask_b = np.stack(mask_list, axis=0)

        self._incoming.put((self._actor_id, self._model_id, obs_b, mask_b))
        while True:
            aid, mid, data = self._result.get()
            if aid == self._actor_id and mid == self._model_id:
                if isinstance(data, Exception):
                    raise data
                break

        return self._process_batch(data, mask_b)

    def _process_batch(self, data, mask_b):
        """Sparse softmax on actor side — only over legal actions (~40 dims)."""
        policy_logits, value_logits = data
        n = len(policy_logits)
        values = np.empty((n, 3), dtype=np.float32)
        prior_list = []
        for i in range(n):
            legal = mask_b[i].nonzero()[0]
            pl = policy_logits[i, legal]
            pl = np.clip(pl, -30, 30)
            pl = np.exp(pl - pl.max())
            probs = (pl / pl.sum()).astype(np.float32)
            prior_list.append(
                list(zip(legal.astype(int).tolist(), probs.astype(float).tolist())))
            vl = value_logits[i]
            vl = np.exp(vl - vl.max())
            values[i] = vl / vl.sum()
        return values, prior_list


# ── GPU Inference Server ────────────────────────────────────────────────────

_MAX_WAIT = 0.010


def _run_server(incoming_q, result_qs, model_specs, game_name, max_batch,
               build_model_fn=None):
    """GPU inference process with multi-model support.

    *model_specs*: dict model_id → {state_dict, nn_width, nn_depth}
    *build_model_fn*: callable(game, cfg_kwargs) → Model (default: build_othello_model)
    """
    import os as _os

    # ── Priority / CPU affinity ───────────────────────────────────────────────
    try:
        if _os.name == "posix":
            _os.nice(-10)
        else:
            import ctypes as _ctypes
            _ctypes.windll.kernel32.SetPriorityClass(
                _ctypes.windll.kernel32.GetCurrentProcess(), 0x00008000)
    except Exception:
        pass  # non-root — try CPU affinity fallback

    try:
        allowed = sorted(_os.sched_getaffinity(0))
        if len(allowed) >= 4:
            # Pin to different core based on process ID to avoid collisions
            core = allowed[-(1 + (_os.getpid() % min(4, len(allowed) - 1)))]
            _os.sched_setaffinity(0, {core})
            print(f"[inference-server] CPU affinity: core {core} "
                  f"(of {len(allowed)} available)", flush=True)
    except Exception as _e:
        print(f"[inference-server] CPU affinity failed: {_e}", flush=True)

    import pyspiel
    if build_model_fn is None:
        from train.core.model_builder import build_othello_model as build_model_fn

    # ── Hardware monitoring (optional) ─────────────────────────────────────
    try:
        import pynvml as _nvml_lib
        _nvml_lib.nvmlInit()
        _nvml_handle = _nvml_lib.nvmlDeviceGetHandleByIndex(0)
        _has_nvml = True
    except Exception:
        try:
            import nvidia_ml_py as _nvml_lib
            _nvml_lib.nvmlInit()
            _nvml_handle = _nvml_lib.nvmlDeviceGetHandleByIndex(0)
            _has_nvml = True
        except Exception:
            _has_nvml = False
    try:
        import psutil
        _has_psutil = True
    except ImportError:
        _has_psutil = False

    game = pyspiel.load_game(game_name)
    models = {}
    for mid, spec in model_specs.items():
        if spec.get("model") is not None:
            models[mid] = spec["model"]
        else:
            m = build_model_fn(game, {
                "nn_width": spec["nn_width"], "nn_depth": spec["nn_depth"],
                "device": "cuda", "path": ".",
            })
            if spec.get("state_dict") is not None:
                m._model.load_state_dict(spec["state_dict"])
            m._model.to("cuda")
            m.eval()
            models[mid] = m

    _cfg = {"max_batch": max_batch}
    _srv_proc = psutil.Process() if _has_psutil else None
    if _srv_proc:
        _srv_proc.cpu_percent()  # warmup: first call returns 0
    int_batches = 0
    int_states = 0
    int_fwd_ms = 0.0
    int_collect_ms = 0.0
    int_scatter_ms = 0.0
    last_report = time.time()
    report_interval = 15.0  # seconds, doubles each report up to 32 min

    # Per-model pending queues with arrival timestamps
    from collections import defaultdict
    pending = defaultdict(list)  # model_id → [(arrival_time, actor_id, obs_list, mask_list), ...]

    while True:
        # ── Collect all incoming messages ──────────────────────────────
        while True:
            try:
                msg = incoming_q.get_nowait()
            except queue.Empty:
                break

            if msg == "__STOP__":
                return

            if isinstance(msg, tuple) and len(msg) == 3 \
                    and isinstance(msg[1], dict):
                mid, sd, mb = msg
                if mid in models and hasattr(models[mid], '_model'):
                    models[mid]._model.load_state_dict(sd)
                _cfg["max_batch"] = mb
                continue

            actor_id, model_id, obs_batch, mask_batch = msg
            if model_id not in models:
                if actor_id in result_qs:
                    result_qs[actor_id].put(
                        (actor_id, model_id,
                         RuntimeError(f"unknown model_id: {model_id}")))
                continue
            pending[model_id].append(
                (time.time(), actor_id, obs_batch, mask_batch))

        # ── Batching: wait for more requests if batch isn't full ────────
        total_states = sum(
            sum(len(r[2]) for r in reqs) for reqs in pending.values())
        oldest_arrival = min(
            (r[0] for reqs in pending.values() for r in reqs),
            default=time.time())
        oldest_ms = (time.time() - oldest_arrival) * 1000

        # Only process when: batch full OR oldest request has waited enough
        BATCH_WAIT_MS = 2.0   # max time to wait for batching
        if total_states < _cfg["max_batch"] and oldest_ms < BATCH_WAIT_MS:
            time.sleep(0.0005)
            continue

        if total_states == 0:
            time.sleep(0.001)
            continue

        # ── Pick model ──────────────────────────────────────────────────
        now = time.time()
        best_score = -1.0
        best_mid = None
        for mid, reqs in pending.items():
            if not reqs:
                continue
            wait_ms = (now - reqs[0][0]) * 1000
            n_states = sum(len(r[2]) for r in reqs)  # obs_batch.shape[0]
            fill = min(n_states / _cfg["max_batch"], 1.0)
            score = wait_ms * (fill ** 0.5)
            if score > best_score:
                best_score = score
                best_mid = mid

        # Collect up to max_batch states from the best model
        group = []
        total_states = 0
        kept = []
        for req in pending[best_mid]:
            _, actor_id, obs_batch, mask_batch = req
            n = len(obs_batch)
            if total_states + n <= _cfg["max_batch"]:
                group.append((actor_id, obs_batch, mask_batch))
                total_states += n
            else:
                kept.append(req)
        pending[best_mid] = kept

        if not group:
            time.sleep(0.001)
            continue

        # ── Forward for this model ────────────────────────────────────
        model = models[best_mid]

        concat_t0 = time.time()
        all_obs = np.concatenate([o for _, o, _ in group], axis=0)
        all_mask = np.concatenate([m for _, _, m in group], axis=0)
        concat_ms = (time.time() - concat_t0) * 1000

        fwd_t0 = time.time()
        policy_logits, value_logits = model.batch_forward_raw(all_obs)
        fwd_ms = (time.time() - fwd_t0) * 1000

        scatter_t0 = time.time()
        cursor = 0
        for actor_id, obs_batch, _ in group:
            n = len(obs_batch)
            result_qs[actor_id].put((
                actor_id, best_mid,
                (policy_logits[cursor:cursor + n],
                 value_logits[cursor:cursor + n])))
            cursor += n
        scatter_ms = (time.time() - scatter_t0) * 1000

        int_batches += 1
        int_states += total_states
        int_fwd_ms += fwd_ms
        int_collect_ms += concat_ms
        int_scatter_ms += scatter_ms

        now = time.time()
        if now - last_report >= report_interval:
            elapsed = now - last_report
            n = max(int_batches, 1)
            hw = ""
            if _has_nvml:
                util = _nvml_lib.nvmlDeviceGetUtilizationRates(_nvml_handle)
                import torch
                vram = torch.cuda.memory_allocated() / (1024 ** 3)
                hw += f"  gpu={util.gpu}% vram={vram:.1f}GB"
            if _has_psutil:
                srv_cpu = _srv_proc.cpu_percent() or 0.0
                # Collect + cache actor Process objects (lazy, first report)
                if "_actor_procs" not in _cfg:
                    _cfg["_actor_procs"] = []
                    _cfg["_actor_pids"] = set()
                    try:
                        for sib in _srv_proc.parent().children():
                            if sib.pid != _srv_proc.pid:
                                sib.cpu_percent()  # warmup
                                _cfg["_actor_procs"].append(sib)
                                _cfg["_actor_pids"].add(sib.pid)
                    except Exception:
                        pass
                else:
                    # Remove dead actors, add new ones
                    try:
                        cur_pids = {c.pid for c in _srv_proc.parent().children()
                                    if c.pid != _srv_proc.pid}
                        new = cur_pids - _cfg["_actor_pids"]
                        for sib in _srv_proc.parent().children():
                            if sib.pid in new:
                                sib.cpu_percent()
                                _cfg["_actor_procs"].append(sib)
                                _cfg["_actor_pids"].add(sib.pid)
                        _cfg["_actor_procs"] = [p for p in _cfg["_actor_procs"]
                                                if p.pid in cur_pids]
                        _cfg["_actor_pids"] &= cur_pids
                    except Exception:
                        pass
                act_cpu = sum(
                    (p.cpu_percent() or 0.0) for p in _cfg["_actor_procs"]
                )
                act_count = len(_cfg["_actor_procs"])
                hw += (f"  cpu_srv={srv_cpu:.0f}%"
                       f"  cpu_act={act_cpu:.0f}%(@{act_count})"
                       f"  mem={psutil.virtual_memory().percent}%")
            print(f"[inference-server] batches={int_batches}  "
                  f"states={int_states}  "
                  f"avg_batch={int_states/n:.1f}  "
                  f"avg_fwd={int_fwd_ms/n:.1f}ms  "
                  f"avg_ipc={int_collect_ms/n:.1f}ms  "
                  f"avg_scatter={int_scatter_ms/n:.1f}ms  "
                  f"states/s={int_states/max(elapsed,0.001):.0f}"
                  f"{hw}",
                  flush=True)
            int_batches = 0
            int_states = 0
            int_fwd_ms = 0.0
            int_collect_ms = 0.0
            int_scatter_ms = 0.0
            last_report = now
            report_interval = min(report_interval * 2, 1920.0)


# ── Lifecycle manager ───────────────────────────────────────────────────────

class InferenceServer:

    def __init__(self, build_model_fn=None):
        self._incoming = mp.Queue(maxsize=500)
        self._result_qs = {}
        self._model_specs = {}
        self._proc = None
        self._build_model_fn = build_model_fn

    @property
    def incoming_queue(self):
        return self._incoming

    def register_actor(self, actor_id: int):
        q = mp.Queue(maxsize=100)
        self._result_qs[actor_id] = q
        return q

    def result_queue(self, actor_id: int):
        return self._result_qs[actor_id]

    def register_model(self, model_id: str, state_dict, nn_width,
                       nn_depth):
        self._model_specs[model_id] = {
            "state_dict": state_dict,
            "nn_width": nn_width, "nn_depth": nn_depth,
        }

    def start(self, game_name, max_batch):
        if not self._model_specs:
            raise RuntimeError("No models registered")
        self._proc = mp.Process(
            target=_run_server,
            args=(self._incoming, self._result_qs, self._model_specs,
                  game_name, max_batch, self._build_model_fn),
            daemon=True,
        )
        self._proc.start()

    def update_weights(self, model_id: str, state_dict, max_batch):
        self._incoming.put((model_id, state_dict, max_batch))

    def stop(self):
        """Graceful stop via sentinel (works for both threads and processes)."""
        self._incoming.put("__STOP__")

    def terminate(self):
        if self._proc and self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=5)
