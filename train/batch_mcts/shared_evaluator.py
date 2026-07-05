"""Shared GPU evaluator for multi-actor training.

Supports multiple models via model_id routing — each actor can use a
different checkpoint without spawning separate GPU processes.

Architecture:
  actor → incoming_q: (actor_id, model_id, obs_list, mask_list)
  Server: groups by model_id, batched forward per group, writes to
          shared ServerShm, notifies each actor with (offset, n)
  actor ← result_qs[actor_id]: (actor_id, model_id, offset, n)
  actor reads slice from ServerShm
"""

import multiprocessing as mp
import queue
import time

import numpy as np


def fast_legal_mask(state, num_actions: int) -> np.ndarray:
    """Numpy bool mask of legal actions — avoids pybind list-of-8100-ints.

    ``state.legal_actions_mask()`` returns a C++ ``std::vector<int>(8100)``
    which pybind11 converts to a Python list of 8100 int objects, only for
    ``np.asarray()`` to read them back.  Using ``legal_actions()`` (~44 ints)
    + numpy indexing is 100–1000× cheaper on the Python side.
    """
    m = np.zeros(num_actions, dtype=bool)
    m[state.legal_actions()] = True
    return m


class SharedEvaluator:
    """BatchMCTS-compatible evaluator backed by a shared GPU process."""

    def __init__(self, game, incoming_q: mp.Queue, result_q: mp.Queue,
                 actor_id: int = 0, model_id: str = "main",
                 actor_shm=None, server_shm=None):
        self._game = game
        self._incoming = incoming_q
        self._result = result_q
        self._actor_id = actor_id
        self._model_id = model_id
        self._shm = actor_shm
        self._use_shm = actor_shm is not None
        self._server_shm = server_shm

    def scalar_value(self, state):
        """Return p0-perspective scalar Q from WDL output."""
        value, _ = self._inference(state)
        q = float(value[0] - value[2])
        if state.current_player() == 1:
            q = -q
        return q

    def _inference(self, state):
        val, prior = self._infer_one(state)
        policy = np.zeros(self._game.num_distinct_actions(), dtype=np.float32)
        for a, p in prior:
            policy[a] = p
        return val, policy

    def _infer_one(self, state):
        obs = np.asarray(state.observation_tensor(), dtype=np.float32)
        mask = fast_legal_mask(state, self._game.num_distinct_actions())
        obs_b = obs.reshape(1, -1)
        mask_b = mask.reshape(1, -1)
        self._send((obs_b, mask_b))
        data = self._recv()
        values, priors = self._process_batch(data, mask_b)
        return values[0], priors[0]

    def batch_inference_raw(self, states):
        if not states:
            return np.array([]), []

        obs_list = [np.asarray(s.observation_tensor(), dtype=np.float32)
                    for s in states]
        num_act = self._game.num_distinct_actions()
        mask_list = [fast_legal_mask(s, num_act) for s in states]
        obs_b = np.stack(obs_list, axis=0)
        mask_b = np.stack(mask_list, axis=0)

        self._send((obs_b, mask_b))
        data = self._recv()
        return self._process_batch(data, mask_b)

    def _send(self, batch):
        obs_b, mask_b = batch
        if self._use_shm:
            self._shm.write_input_obs(obs_b)
            self._incoming.put((self._actor_id, self._model_id,
                                "shm", obs_b.shape[0]))
        else:
            self._incoming.put((self._actor_id, self._model_id,
                                obs_b, mask_b))

    def _recv(self):
        while True:
            try:
                msg = self._result.get(timeout=60)
            except queue.Empty:
                raise RuntimeError(
                    f"SharedEvaluator actor={self._actor_id}: "
                    f"no response from server after 60s — server may have crashed")
            aid, mid = msg[:2]
            if aid == self._actor_id and mid == self._model_id:
                if len(msg) == 4:
                    # SHM offset protocol: (actor_id, mid, offset, n)
                    offset, n = msg[2], msg[3]
                    return self._server_shm.read_slice(offset, n)
                if len(msg) >= 3 and isinstance(msg[2], Exception):
                    raise msg[2]
                # legacy / non-SHM: (actor_id, model_id, (pl, vl))
                return msg[2]


    def _process_batch(self, data, mask_b):
        """Sparse softmax on actor side — only over legal actions (~40 dims)."""
        policy_logits, value_logits = data
        n = len(policy_logits)
        values = np.empty((n, 3), dtype=np.float32)
        prior_list = []
        for i in range(n):
            legal = mask_b[i].nonzero()[0]
            if len(legal) == 0:
                print(f"[WARN] empty legal mask at idx {i}/{n}", flush=True)
                prior_list.append([])
                continue
            pl = policy_logits[i, legal]
            pl = np.clip(np.nan_to_num(pl, nan=0.0, posinf=30.0, neginf=-30.0), -30, 30)
            pl = np.exp(pl - pl.max())
            probs = np.clip(pl / pl.sum(), 0, 1).astype(np.float32)
            prior_list.append(
                list(zip(legal.astype(int).tolist(), probs.astype(float).tolist())))
            vl = value_logits[i]
            vl = np.exp(vl - vl.max())
            values[i] = vl / vl.sum()
        return values, prior_list


# ── GPU Inference Server ────────────────────────────────────────────────────

_MAX_WAIT = 0.010  # unused, kept for reference


def _run_server(incoming_q, result_qs, model_specs, game_name, max_batch,
               build_model_fn=None, actor_shms=None, server_shm=None,
               gpu_id=0):
    """GPU inference process with multi-model support.

    *actor_shms*: dict actor_id → ActorShm (input buffers).
    *server_shm*: ServerShm — single shared output buffer for all actors.
    """
    import os as _os

    # ── Bind to specific GPU ────────────────────────────────────────────────
    import torch as _torch
    if _torch.cuda.is_available():
        try:
            _torch.cuda.set_device(gpu_id)
        except RuntimeError:
            print(f"[inf-srv GPU{gpu_id}] set_device failed, using default GPU",
                  flush=True)

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
            # Pin to distinct core per GPU (high end; actors use the rest)
            core = allowed[-(1 + gpu_id)]
            _os.sched_setaffinity(0, {core})
            print(f"[inf-srv GPU{gpu_id}] CPU affinity: core {core} "
                  f"(of {len(allowed)} available)", flush=True)
    except Exception as _e:
        print(f"[inf-srv GPU{gpu_id}] CPU affinity failed: {_e}", flush=True)

    import pyspiel
    if build_model_fn is None:
        from train.core.model_builder import build_othello_model as build_model_fn

    # ── Hardware monitoring (optional) ─────────────────────────────────────
    try:
        import pynvml as _nvml_lib
        _nvml_lib.nvmlInit()
        _nvml_handle = _nvml_lib.nvmlDeviceGetHandleByIndex(gpu_id)
        _has_nvml = True
    except Exception:
        try:
            import nvidia_ml_py as _nvml_lib
            _nvml_lib.nvmlInit()
            _nvml_handle = _nvml_lib.nvmlDeviceGetHandleByIndex(gpu_id)
            _has_nvml = True
        except Exception:
            _has_nvml = False
    try:
        import psutil
        _has_psutil = True
    except ImportError:
        _has_psutil = False

    game = pyspiel.load_game(game_name)
    obs_dim = int(np.prod(game.observation_tensor_shape()))
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

    _actor_shms = actor_shms or {}
    _cfg = {"max_batch": max_batch,
            "obs_buf": np.empty((max_batch, obs_dim), dtype=np.float32)}
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
    pending = defaultdict(list)

    def _req_n(req):
        return req[4] if req[4] is not None else len(req[2])

    def _total_pending():
        return sum(sum(_req_n(r) for r in reqs) for reqs in pending.values())

    def _oldest_ms():
        if not any(pending.values()):
            return 0.0
        return (time.time() - min(r[0] for reqs in pending.values()
                                  for r in reqs)) * 1000

    def _handle_msg(msg):
        """Process one incoming message. Returns True if server should stop."""
        if msg == "__STOP__":
            return True

        if isinstance(msg, tuple) and len(msg) == 3 \
                and isinstance(msg[1], dict):
            mid, sd, mb = msg
            if mid in models:
                target = getattr(models[mid], '_model', models[mid])
                if hasattr(target, 'load_state_dict'):
                    target.load_state_dict(sd)
            if mb != _cfg["max_batch"]:
                _cfg["max_batch"] = mb
                _cfg["obs_buf"] = np.empty((mb, obs_dim), dtype=np.float32)
            return False

        is_shm = (len(msg) == 4 and isinstance(msg[2], str)
                  and msg[2] == "shm")
        if is_shm:
            actor_id, model_id, _, n_states = msg
            obs_batch, mask_batch = None, None
        else:
            actor_id, model_id, obs_batch, mask_batch = msg
            n_states = None
        if model_id not in models:
            if actor_id in result_qs:
                result_qs[actor_id].put(
                    (actor_id, model_id,
                     RuntimeError(f"unknown model_id: {model_id}")))
            return False
        pending[model_id].append(
            (time.time(), actor_id, obs_batch, mask_batch, n_states))
        return False

    while True:
        # ── Drain all queued messages (non-blocking) ──────────────────
        while True:
            try:
                msg = incoming_q.get_nowait()
            except queue.Empty:
                break
            if _handle_msg(msg):
                return

        # ── Decide next action ─────────────────────────────────────────
        total = _total_pending()
        if total == 0:
            # Idle: block until a message arrives (no CPU spinning)
            try:
                msg = incoming_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if _handle_msg(msg):
                return
            continue  # drain remaining + re-check

        if total < _cfg["max_batch"] and _oldest_ms() < 0.5:
            # Batch not full: short block for more messages
            try:
                msg = incoming_q.get(timeout=0.0005)
            except queue.Empty:
                continue  # timeout → re-check conditions
            if _handle_msg(msg):
                return
            continue  # drain remaining + re-check

        # ── Batch ready: pick model ────────────────────────────────────
        now = time.time()
        best_score = -1.0
        best_mid = None
        for mid, reqs in pending.items():
            if not reqs:
                continue
            wait_ms = (now - reqs[0][0]) * 1000
            n_s = sum(_req_n(r) for r in reqs)
            fill = min(n_s / _cfg["max_batch"], 1.0)
            score = wait_ms * (fill ** 0.5)
            if score > best_score:
                best_score = score
                best_mid = mid

        # Collect up to max_batch states from the best model
        group = []
        total_states = 0
        kept = []
        for req in pending[best_mid]:
            _, actor_id, obs_batch, mask_batch, n_shm = req
            n = n_shm if n_shm is not None else len(obs_batch)
            can_take = total_states + n <= _cfg["max_batch"]
            if can_take:
                group.append((actor_id, obs_batch, mask_batch, n_shm))
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
        obs_buf = _cfg["obs_buf"]
        resolved = []
        cursor = 0
        for actor_id, obs_batch, mask_batch, n_shm in group:
            if n_shm is not None:
                shm_buf = _actor_shms.get(actor_id)
                obs_b = shm_buf.read_input_obs() if shm_buf else None
                n = n_shm
                if obs_b is not None:
                    obs_buf[cursor:cursor + n] = obs_b
                resolved.append((actor_id, None, None, n, True))
            else:
                n = len(obs_batch)
                obs_buf[cursor:cursor + n] = obs_batch
                resolved.append((actor_id, obs_batch, mask_batch, n, False))
            cursor += n
        concat_ms = (time.time() - concat_t0) * 1000

        fwd_t0 = time.time()
        policy_logits, value_logits = model.batch_forward_raw(obs_buf[:cursor])
        fwd_ms = (time.time() - fwd_t0) * 1000

        scatter_t0 = time.time()

        # Write batch once to shared output SHM
        if server_shm is not None:
            server_shm.write_batch(policy_logits, value_logits)

        cursor = 0
        for actor_id, _, _, n, is_shm in resolved:
            if is_shm:
                result_qs[actor_id].put((actor_id, best_mid, cursor, n))
            else:
                pl = policy_logits[cursor:cursor + n]
                vl = value_logits[cursor:cursor + n]
                result_qs[actor_id].put(
                    (actor_id, best_mid, (pl, vl)))
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
            print(f"[inf-srv GPU{gpu_id}] batches={int_batches}  "
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

    def __init__(self, build_model_fn=None, gpu_id=0):
        self._incoming = mp.Queue(maxsize=500)
        self._result_qs = {}
        self._model_specs = {}
        self._proc = None
        self._build_model_fn = build_model_fn
        self._gpu_id = gpu_id
        self._shm_bufs = {}      # actor_id → ActorShm (input)
        self._shm_names = {}     # actor_id → name
        self._server_shm = None  # ServerShm (output, shared by all actors)
        self._server_shm_name = None
        self._pol_flat = None    # stored from first register_actor

    @property
    def incoming_queue(self):
        return self._incoming

    def register_actor(self, actor_id: int, max_states: int = 0,
                       obs_flat: int = 0, mask_flat: int = 0,
                       pol_flat: int = 0):
        q = mp.Queue(maxsize=100)
        self._result_qs[actor_id] = q
        if max_states > 0:
            from train.batch_mcts.shm import ActorShm
            import os as _os, time as _time
            name = f"jq_{_os.getpid()}_{int(_time.monotonic()*1e6)}_{self._gpu_id}_{actor_id}"
            shm = ActorShm(name, max_states, obs_flat, mask_flat, create=True)
            self._shm_bufs[actor_id] = shm
            self._shm_names[actor_id] = name
            if self._pol_flat is None:
                self._pol_flat = pol_flat
        return q

    def actor_shm_name(self, actor_id: int):
        return self._shm_names.get(actor_id)

    @property
    def server_shm_name(self):
        """Name of the shared output buffer — all actors open this."""
        return self._server_shm_name

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
        # Create shared output SHM (one per GPU, sized for inference_batch_size)
        if self._pol_flat is not None and self._server_shm is None:
            from train.batch_mcts.shm import ServerShm
            import os as _os, time as _time
            pid = _os.getpid()
            ts = int(_time.monotonic() * 1e6)
            out_name = f"jq_out_{pid}_{ts}_{self._gpu_id}"
            self._server_shm = ServerShm(out_name, max_batch, self._pol_flat,
                                         create=True)
            self._server_shm_name = out_name
        self._proc = mp.Process(
            target=_run_server,
            args=(self._incoming, self._result_qs, self._model_specs,
                  game_name, max_batch, self._build_model_fn,
                  self._shm_bufs, self._server_shm, self._gpu_id),
            daemon=True,
        )
        self._proc.start()

    def shutdown(self):
        """Clean up shared memory buffers."""
        for shm in self._shm_bufs.values():
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass
        self._shm_bufs.clear()
        self._shm_names.clear()
        if self._server_shm is not None:
            try:
                self._server_shm.close()
                self._server_shm.unlink()
            except Exception:
                pass
            self._server_shm = None
            self._server_shm_name = None

    def update_weights(self, model_id: str, state_dict, max_batch):
        self._incoming.put((model_id, state_dict, max_batch))

    def stop(self):
        """Graceful stop via sentinel (works for both threads and processes)."""
        self._incoming.put("__STOP__")

    def terminate(self):
        if self._proc and self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=5)
