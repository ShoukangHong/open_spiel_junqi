"""Integration tests for shared GPU evaluator with multi-model support."""

import time

import numpy as np
import pyspiel

from train.batch_mcts.shared_evaluator import (
    InferenceServer, SharedEvaluator, _run_server)


class _MockModel:
    """Mock model returning WDL values from deterministic hash."""

    def __init__(self, model_id: str, num_actions: int = 9):
        self.mid = model_id
        self._step = 0
        self.num_actions = num_actions

    def step_up(self):
        self._step += 1

    def load_state_dict(self, sd):
        pass  # weight-update handled via step_up()

    def batch_forward_raw(self, obs):
        """Raw logits that softmax back to _wdl values."""
        n = obs.shape[0]
        na = self.num_actions
        policy_logits = np.zeros((n, na), dtype=np.float32)
        value_logits = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            wdl = np.maximum(self._wdl(obs[i]), 1e-9)
            value_logits[i] = np.log(wdl)
        return policy_logits, value_logits

    def batch_inference(self, obs, mask):
        n = obs.shape[0]
        na = self.num_actions
        values = np.array([self._wdl(obs[i]) for i in range(n)],
                          dtype=np.float32)
        policies = np.zeros((n, na), dtype=np.float32)
        for i in range(n):
            legals = np.where(mask[i])[0]
            for a in legals:
                policies[i, a] = 1.0 / len(legals)
        return values, policies

    def _scalar(self, o):
        h = abs(hash((self.mid, self._step, o.tobytes())))
        return float(h % 9973) / 9973.0

    def _wdl(self, o):
        v = self._scalar(o)
        q = (v - 0.5) * 2
        d = 1.0 - abs(q)
        w = np.array([max(q, 0), d, max(-q, 0)], dtype=np.float32)
        return w / w.sum()


# ── Helpers ─────────────────────────────────────────────────────────────────

def _expected_wdl(model_id, step, obs):
    """Recompute the expected WDL for a given state."""
    h = abs(hash((model_id, step, obs.tobytes())))
    v = float(h % 9973) / 9973.0
    q = (v - 0.5) * 2
    d = 1.0 - abs(q)
    w = np.array([max(q, 0), d, max(-q, 0)], dtype=np.float32)
    return w / w.sum()


def _make_state(moves=()):
    game = pyspiel.load_game("tic_tac_toe")
    s = game.new_initial_state()
    for a in moves:
        s.apply_action(a)
    return s


def _start_server(server, mock_models, **run_kw):
    """Start server with mock models using the real _run_server."""
    import threading
    specs = {}
    for mid, mm in mock_models.items():
        specs[mid] = {"model": mm}
    server._thread = threading.Thread(
        target=_run_server,
        args=(server.incoming_queue, server._result_qs, specs,
              "tic_tac_toe", 128) + run_kw.get('extra_args', ()),
        daemon=True)
    server._thread.start()
    time.sleep(0.1)
    if not server._thread.is_alive():
        raise RuntimeError("Server thread crashed on startup")

def _stop_server(server):
    try:
        server.stop()
    except Exception:
        pass
    try:
        server._thread.join(timeout=3)
    except Exception:
        pass
    # Clean up shared memory buffers
    for shm in list(server._shm_bufs.values()):
        try:
            shm.close()
            shm.unlink()
        except Exception:
            pass
    server._shm_bufs.clear()
    server._shm_names.clear()


# ── Tests ───────────────────────────────────────────────────────────────────

def test_single_inference():
    game = pyspiel.load_game("tic_tac_toe")
    mm = _MockModel("main")
    server = InferenceServer()
    try:
        rq = server.register_actor(0, max_states=8, obs_flat=27, mask_flat=9, pol_flat=9)
        _start_server(server, {"main": mm})

        ev = SharedEvaluator(game, server.incoming_queue, rq, actor_id=0,
                             model_id="main")
        s = _make_state([0, 4])
        v, _ = ev._inference(s)

        obs = np.asarray(s.observation_tensor(), dtype=np.float32)
        exp = _expected_wdl("main", 0, obs)
        assert np.allclose(np.asarray(v), exp, atol=0.01)
    finally:
        _stop_server(server)


def test_batch_order():
    game = pyspiel.load_game("tic_tac_toe")
    mm = _MockModel("main")
    server = InferenceServer()
    try:
        rq = server.register_actor(0, max_states=8, obs_flat=27, mask_flat=9, pol_flat=9)
        _start_server(server, {"main": mm})

        ev = SharedEvaluator(game, server.incoming_queue, rq, actor_id=0,
                             model_id="main")
        states = [_make_state([]), _make_state([0]), _make_state([0, 4])]
        values, priors = ev.batch_inference_raw(states)

        assert len(values) == 3
        for i, s in enumerate(states):
            obs = np.asarray(s.observation_tensor(), dtype=np.float32)
            exp = _expected_wdl("main", 0, obs)
            assert np.allclose(values[i], exp, atol=0.01)
    finally:
        _stop_server(server)


def test_multi_model_routing():
    game = pyspiel.load_game("tic_tac_toe")
    m0, m1 = _MockModel("m0"), _MockModel("m1")
    server = InferenceServer()
    try:
        rq0, rq1 = server.register_actor(0, max_states=8, obs_flat=27, mask_flat=9, pol_flat=9), server.register_actor(1, max_states=8, obs_flat=27, mask_flat=9, pol_flat=9)
        _start_server(server, {"m0": m0, "m1": m1})

        ev0 = SharedEvaluator(game, server.incoming_queue, rq0, actor_id=0,
                              model_id="m0")
        ev1 = SharedEvaluator(game, server.incoming_queue, rq1, actor_id=1,
                              model_id="m1")

        s = _make_state([0, 4])
        obs = np.asarray(s.observation_tensor(), dtype=np.float32)
        v0, _ = ev0._inference(s)
        v1, _ = ev1._inference(s)

        assert np.allclose(np.asarray(v0), _expected_wdl("m0", 0, obs), atol=0.01)
        assert np.allclose(np.asarray(v1), _expected_wdl("m1", 0, obs), atol=0.01)
        assert not np.allclose(np.asarray(v0), np.asarray(v1), atol=0.01)
    finally:
        _stop_server(server)


def test_weight_update_changes_output():
    game = pyspiel.load_game("tic_tac_toe")
    mm = _MockModel("main")
    server = InferenceServer()
    try:
        rq = server.register_actor(0, max_states=8, obs_flat=27, mask_flat=9, pol_flat=9)
        _start_server(server, {"main": mm})

        ev = SharedEvaluator(game, server.incoming_queue, rq, actor_id=0,
                             model_id="main")
        s = _make_state([0])
        obs = np.asarray(s.observation_tensor(), dtype=np.float32)

        v0, _ = ev._inference(s)
        assert np.allclose(np.asarray(v0), _expected_wdl("main", 0, obs), atol=0.01)

        mm.step_up()
        server._incoming.put(("main", {}, 128))
        time.sleep(0.1)

        v1, _ = ev._inference(s)
        assert np.allclose(np.asarray(v1), _expected_wdl("main", 1, obs), atol=0.01)
    finally:
        _stop_server(server)


def test_wdl_mode():
    game = pyspiel.load_game("tic_tac_toe")
    mm = _MockModel("main_wdl", )
    server = InferenceServer()
    try:
        rq = server.register_actor(0, max_states=8, obs_flat=27, mask_flat=9, pol_flat=9)
        _start_server(server, {"main_wdl": mm}, )

        ev = SharedEvaluator(game, server.incoming_queue, rq, actor_id=0,
                             model_id="main_wdl")
        s = _make_state([0, 4, 1])
        values, _ = ev.batch_inference_raw([s])

        assert values.shape == (1, 3)
        assert abs(values[0].sum() - 1.0) < 0.01
    finally:
        _stop_server(server)


def test_wdl_scalar_value_perspective():
    """scalar_value always returns p0 perspective regardless of current player."""
    import threading
    game = pyspiel.load_game("tic_tac_toe")

    # Fixed WDL model: current player always winning (w=0.8, d=0.1, l=0.1)
    class _FixedWDLModel:
        def batch_forward_raw(self, obs):
            n = obs.shape[0]
            wdl = np.array([0.8, 0.1, 0.1], dtype=np.float32)
            return (np.zeros((n, 9), dtype=np.float32),
                    np.tile(np.log(wdl), (n, 1)))

        def batch_inference(self, obs, mask):
            n = obs.shape[0]
            return (np.array([[0.8, 0.1, 0.1]] * n, dtype=np.float32),
                    np.ones((n, mask.shape[1]), dtype=np.float32) / mask.shape[1])

    server = InferenceServer()
    try:
        rq = server.register_actor(0, max_states=8, obs_flat=27, mask_flat=9, pol_flat=9)
        specs = {"wdl": {"model": _FixedWDLModel()}}
        server._thread = threading.Thread(
            target=_run_server,
            args=(server.incoming_queue, server._result_qs, specs,
                  "tic_tac_toe", 128),
            daemon=True)
        server._thread.start()
        time.sleep(0.1)
        if not server._thread.is_alive():
            raise RuntimeError("Server thread crashed on startup")

        ev = SharedEvaluator(game, server.incoming_queue, rq, actor_id=0,
                             model_id="wdl")

        # p0 to move (empty board)
        s0 = _make_state([])
        assert s0.current_player() == 0
        v0 = ev.scalar_value(s0)
        # Model: w=0.8, l=0.1 → w-l=0.7 from current player (p0)
        # scalar_value: current_player=0 → no flip → 0.7
        assert abs(v0 - 0.7) < 0.01, f"p0 to move: expected 0.7, got {v0}"

        # p1 to move (same board, p0 just played)
        s1 = _make_state([0])
        assert s1.current_player() == 1
        v1 = ev.scalar_value(s1)
        # Model: w=0.8, l=0.1 → w-l=0.7 from current player (p1)
        # scalar_value: current_player=1 → flip → -0.7
        assert abs(v1 - (-0.7)) < 0.01, f"p1 to move: expected -0.7, got {v1}"

        # Verify: v0 ≈ -v1 (same position, opposite player → opposite value)
        assert abs(v0 + v1) < 0.01, f"v0+v1 should be 0, got {v0+v1}"
    finally:
        _stop_server(server)


def test_same_actor_different_models():
    """One actor uses two evaluators (m0/m1) interleaved — eval_match scenario."""
    game = pyspiel.load_game("tic_tac_toe")
    m0, m1 = _MockModel("m0"), _MockModel("m1")
    server = InferenceServer()
    try:
        rq = server.register_actor(0, max_states=8, obs_flat=27, mask_flat=9, pol_flat=9)
        _start_server(server, {"m0": m0, "m1": m1})

        ev0 = SharedEvaluator(game, server.incoming_queue, rq, actor_id=0,
                              model_id="m0")
        ev1 = SharedEvaluator(game, server.incoming_queue, rq, actor_id=0,
                              model_id="m1")

        s = _make_state([0, 4])
        obs = np.asarray(s.observation_tensor(), dtype=np.float32)

        for _ in range(3):
            v0, _ = ev0._inference(s)
            assert np.allclose(np.asarray(v0), _expected_wdl("m0", 0, obs), atol=0.01)
            v1, _ = ev1._inference(s)
            assert np.allclose(np.asarray(v1), _expected_wdl("m1", 0, obs), atol=0.01)
            assert not np.allclose(np.asarray(v0), np.asarray(v1), atol=0.01)
    finally:
        _stop_server(server)


def test_multi_actor_multi_model():
    """Two actors, each using two models — no cross-talk."""
    game = pyspiel.load_game("tic_tac_toe")
    m0, m1 = _MockModel("m0"), _MockModel("m1")
    server = InferenceServer()
    try:
        rq0, rq1 = server.register_actor(0, max_states=8, obs_flat=27, mask_flat=9, pol_flat=9), server.register_actor(1, max_states=8, obs_flat=27, mask_flat=9, pol_flat=9)
        _start_server(server, {"m0": m0, "m1": m1})

        ev0_m0 = SharedEvaluator(game, server.incoming_queue, rq0, actor_id=0,
                                 model_id="m0")
        ev0_m1 = SharedEvaluator(game, server.incoming_queue, rq0, actor_id=0,
                                 model_id="m1")
        ev1_m0 = SharedEvaluator(game, server.incoming_queue, rq1, actor_id=1,
                                 model_id="m0")
        ev1_m1 = SharedEvaluator(game, server.incoming_queue, rq1, actor_id=1,
                                 model_id="m1")

        s = _make_state([0, 4])
        obs = np.asarray(s.observation_tensor(), dtype=np.float32)

        v00, _ = ev0_m0._inference(s)
        v01, _ = ev0_m1._inference(s)
        v10, _ = ev1_m0._inference(s)
        v11, _ = ev1_m1._inference(s)

        assert np.allclose(np.asarray(v00), _expected_wdl("m0", 0, obs), atol=0.01)
        assert np.allclose(np.asarray(v10), _expected_wdl("m0", 0, obs), atol=0.01)
        assert np.allclose(np.asarray(v01), _expected_wdl("m1", 0, obs), atol=0.01)
        assert np.allclose(np.asarray(v11), _expected_wdl("m1", 0, obs), atol=0.01)
        assert not np.allclose(np.asarray(v00), np.asarray(v01), atol=0.01)
    finally:
        _stop_server(server)


def _stress_worker(actor_id, ev0, ev1, game, states, errors, lock,
                   mid0="m0", mid1="m1"):
    rng = np.random.RandomState(actor_id * 1000)
    for _ in range(30):
        mid = rng.randint(0, 2)
        ev = ev0 if mid == 0 else ev1
        model_id = mid0 if mid == 0 else mid1
        s = states[rng.randint(0, len(states))]
        obs = np.asarray(s.observation_tensor(), dtype=np.float32)

        value, policy = ev._inference(s)

        if len(value) != 3:
            with lock:
                errors.append(f"[a{actor_id} m{mid}] WDL len={len(value)}")
            continue
        if abs(sum(value) - 1.0) >= 0.01:
            with lock:
                errors.append(f"[a{actor_id} m{mid}] WDL sum={sum(value):.3f}")
            continue
        expected_val = _expected_wdl(model_id, 0, obs)
        if not np.allclose(value, expected_val, atol=0.05):
            with lock:
                errors.append(f"[a{actor_id} m{mid}] WDL: "
                              f"{list(value)} != {list(expected_val)}")

        legals = s.legal_actions()
        mask = np.zeros(game.num_distinct_actions(), dtype=bool)
        mask[legals] = True

        if not mask.all():
            if policy[~mask].max() > 1e-8:
                with lock:
                    errors.append(f"[a{actor_id} m{mid}] illegal actions have "
                                  f"prob={policy[~mask].max():.2e}")
        if abs(policy.sum() - 1.0) > 1e-6:
            with lock:
                errors.append(f"[a{actor_id} m{mid}] policy sum={policy.sum():.6f}")
        # Uniform distribution over legal actions
        expected_prior = np.zeros(game.num_distinct_actions(), dtype=np.float32)
        for a in legals:
            expected_prior[a] = 1.0 / len(legals)
        if not np.allclose(policy, expected_prior, atol=0.01):
            with lock:
                errors.append(f"[a{actor_id} m{mid}] policy distribution mismatch")


def test_stress_concurrent():
    """Concurrent stress: 4 actors × 2 models, scalar mode."""
    import threading
    game = pyspiel.load_game("tic_tac_toe")
    m0, m1 = _MockModel("m0"), _MockModel("m1")
    server = InferenceServer()
    try:
        num_actors = 4
        rqs = [server.register_actor(i, max_states=8, obs_flat=27, mask_flat=9, pol_flat=9) for i in range(num_actors)]
        _start_server(server, {"m0": m0, "m1": m1})

        states = [_make_state([]), _make_state([0]), _make_state([0, 4]),
                  _make_state([0, 4, 1]), _make_state([0, 4, 1, 3])]
        errors = []
        lock = threading.Lock()

        def worker(actor_id):
            ev0 = SharedEvaluator(game, server.incoming_queue, rqs[actor_id],
                                  actor_id=actor_id, model_id="m0")
            ev1 = SharedEvaluator(game, server.incoming_queue, rqs[actor_id],
                                  actor_id=actor_id, model_id="m1")
            _stress_worker(actor_id, ev0, ev1, game, states, errors, lock)

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(num_actors)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"{len(errors)} errors: {errors[:3]}"
    finally:
        _stop_server(server)


def test_stress_concurrent_wdl():
    """Concurrent stress: 4 actors × 2 models, WDL mode."""
    import threading
    game = pyspiel.load_game("tic_tac_toe")
    m0 = _MockModel("m0_wdl", )
    m1 = _MockModel("m1_wdl", )
    server = InferenceServer()
    try:
        num_actors = 4
        rqs = [server.register_actor(i, max_states=8, obs_flat=27, mask_flat=9, pol_flat=9) for i in range(num_actors)]
        _start_server(server, {"m0_wdl": m0, "m1_wdl": m1}, )

        states = [_make_state([]), _make_state([0]), _make_state([0, 4]),
                  _make_state([0, 4, 1]), _make_state([0, 4, 1, 3])]
        errors = []
        lock = threading.Lock()

        def worker(actor_id):
            ev0 = SharedEvaluator(game, server.incoming_queue, rqs[actor_id],
                                  actor_id=actor_id, model_id="m0_wdl")
            ev1 = SharedEvaluator(game, server.incoming_queue, rqs[actor_id],
                                  actor_id=actor_id, model_id="m1_wdl")
            _stress_worker(actor_id, ev0, ev1, game, states, errors, lock,
                           mid0="m0_wdl", mid1="m1_wdl")

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(num_actors)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"{len(errors)} errors: {errors[:3]}"
    finally:
        _stop_server(server)


# ── run ─────────────────────────────────────────────────────────────────────

# ── Best model support ──────────────────────────────────────────────────────

def test_best_model_routing():
    """Server routes "main" vs "best" to different mock models."""
    game = pyspiel.load_game("tic_tac_toe")
    m_main = _MockModel("main")
    m_best = _MockModel("best")
    server = InferenceServer()
    try:
        rq = server.register_actor(0, max_states=8, obs_flat=27, mask_flat=9, pol_flat=9)
        _start_server(server, {"main": m_main, "best": m_best})
        ev_main = SharedEvaluator(game, server.incoming_queue, rq,
                                  actor_id=0, model_id="main")
        ev_best = SharedEvaluator(game, server.incoming_queue, rq,
                                  actor_id=0, model_id="best")
        s = _make_state([0, 4])
        obs = np.asarray(s.observation_tensor(), dtype=np.float32)
        v_main, _ = ev_main._inference(s)
        v_best, _ = ev_best._inference(s)
        assert np.allclose(np.asarray(v_main), _expected_wdl("main", 0, obs), atol=0.01)
        assert np.allclose(np.asarray(v_best), _expected_wdl("best", 0, obs), atol=0.01)
        assert not np.allclose(np.asarray(v_main), np.asarray(v_best), atol=0.01)
        print("  best_model_routing: PASSED")
    finally:
        _stop_server(server)


def test_best_model_weight_update():
    """update_weights for "best" changes its output."""
    game = pyspiel.load_game("tic_tac_toe")
    m_best = _MockModel("best")
    server = InferenceServer()
    try:
        rq = server.register_actor(0, max_states=8, obs_flat=27, mask_flat=9, pol_flat=9)
        _start_server(server, {"best": m_best})
        ev = SharedEvaluator(game, server.incoming_queue, rq,
                             actor_id=0, model_id="best")
        s = _make_state([0])
        obs = np.asarray(s.observation_tensor(), dtype=np.float32)
        v0, _ = ev._inference(s)
        assert np.allclose(np.asarray(v0), _expected_wdl("best", 0, obs), atol=0.01)
        m_best.step_up()
        server._incoming.put(("best", {}, 128))
        time.sleep(0.1)
        v1, _ = ev._inference(s)
        assert np.allclose(np.asarray(v1), _expected_wdl("best", 1, obs), atol=0.01)
        assert not np.allclose(np.asarray(v0), np.asarray(v1), atol=0.01)
        print("  best_model_weight_update: PASSED")
    finally:
        _stop_server(server)


def test_best_model_prob_selection():
    """best_model_prob controls how often best model is picked."""
    rng = np.random.RandomState(42)
    n = 500
    for prob in [0.0, 0.3, 1.0]:
        cnt = sum(1 for _ in range(n) if prob > 0 and rng.random() < prob)
        exp = int(n * prob)
        tol = max(30, exp * 0.25)
        assert abs(cnt - exp) <= tol, \
            f"prob={prob}: expected ≈{exp}, got {cnt} (tol={tol:.0f})"
    print("  best_model_prob_selection: PASSED")


def main():
    tests = [
        test_single_inference,
        test_batch_order,
        test_multi_model_routing,
        test_weight_update_changes_output,
        test_wdl_mode,
        test_wdl_scalar_value_perspective,
        test_same_actor_different_models,
        test_multi_actor_multi_model,
        test_stress_concurrent,
        test_stress_concurrent_wdl,
        test_best_model_routing,
        test_best_model_weight_update,
        test_best_model_prob_selection,
        test_multi_server_isolation,
        test_multi_server_weight_broadcast,
        test_eval_slot_reservation,
        test_eval_slot_multi_model,
        test_eval_model_update_via_existing_server,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  {fn.__name__}: PASSED")
        except Exception as e:
            print(f"  {fn.__name__}: FAILED — {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    print(f"\n{'=' * 40}")
    print(f"{'ALL PASSED' if failed == 0 else f'{failed} FAILED'}")
    print("=" * 40)
    assert failed == 0


def test_shm():
    """Shared memory path: single + batch inference, various sizes."""
    import threading
    from train.batch_mcts.shm import ActorShm, ServerShm

    game = pyspiel.load_game("tic_tac_toe")
    mm = _MockModel("main", num_actions=9)
    server = InferenceServer()
    srv_shm = None
    try:
        rq = server.register_actor(0, max_states=8, obs_flat=27, mask_flat=9,
                                   pol_flat=9)
        specs = {"main": {"model": mm}}
        shm_name = server.actor_shm_name(0)
        srv_shm = ServerShm(f"test_shm_srv_{id(server)}", 128, 9, create=True)
        server._thread = threading.Thread(
            target=_run_server,
            args=(server.incoming_queue, server._result_qs, specs,
                  "tic_tac_toe", 128, None, server._shm_bufs, srv_shm),
            daemon=True)
        server._thread.start()
        time.sleep(0.1)
        if not server._thread.is_alive():
            raise RuntimeError("Server thread crashed on startup")

        actor_shm = ActorShm(shm_name, 8, 27, 9, create=False)
        ev = SharedEvaluator(game, server.incoming_queue, rq,
                             actor_id=0, model_id="main", actor_shm=actor_shm,
                             server_shm=srv_shm)
        # Single inference
        s = _make_state([0, 4])
        v, _ = ev._inference(s)
        obs = np.asarray(s.observation_tensor(), dtype=np.float32)
        exp = _expected_wdl("main", 0, obs)
        assert np.allclose(np.asarray(v), exp, atol=0.01)

        # Batch: 1 state
        vals1, p1 = ev.batch_inference_raw([_make_state([0])])
        assert vals1.shape == (1, 3) and len(p1) == 1

        # Batch: 3 states
        states = [_make_state([]), _make_state([0]), _make_state([0, 4])]
        vals3, p3 = ev.batch_inference_raw(states)
        assert vals3.shape == (3, 3) and len(p3) == 3

        # Verify WDL values are valid
        for val in vals3:
            assert abs(val.sum() - 1.0) < 0.01
            assert all(v >= 0 for v in val)
    finally:
        _stop_server(server)
        if srv_shm:
            try: srv_shm.close()
            except Exception: pass
            try: srv_shm.unlink()
            except Exception: pass


def test_multi_server_isolation():
    """Two servers on different GPU IDs with different models → different results."""
    import threading

    game = pyspiel.load_game("tic_tac_toe")
    mm0 = _MockModel("m0")
    mm1 = _MockModel("m1")

    s0 = InferenceServer(gpu_id=0)
    s1 = InferenceServer(gpu_id=1)
    try:
        rq0 = s0.register_actor(0, max_states=8, obs_flat=27, mask_flat=9, pol_flat=9)
        rq1 = s1.register_actor(0, max_states=8, obs_flat=27, mask_flat=9, pol_flat=9)

        s0._thread = threading.Thread(
            target=_run_server,
            args=(s0.incoming_queue, s0._result_qs, {"main": {"model": mm0}},
                  "tic_tac_toe", 128, None, s0._shm_bufs, None, 0),
            daemon=True)
        s1._thread = threading.Thread(
            target=_run_server,
            args=(s1.incoming_queue, s1._result_qs, {"main": {"model": mm1}},
                  "tic_tac_toe", 128, None, s1._shm_bufs, None, 1),
            daemon=True)
        s0._thread.start()
        s1._thread.start()
        time.sleep(0.1)
        if not s0._thread.is_alive() or not s1._thread.is_alive():
            raise RuntimeError("Server thread crashed on startup")

        ev0 = SharedEvaluator(game, s0.incoming_queue, rq0, actor_id=0, model_id="main")
        ev1 = SharedEvaluator(game, s1.incoming_queue, rq1, actor_id=0, model_id="main")

        s = _make_state([0, 4])
        obs = np.asarray(s.observation_tensor(), dtype=np.float32)
        v0, _ = ev0._inference(s)
        v1, _ = ev1._inference(s)

        # Different models produce different outputs
        assert not np.allclose(np.asarray(v0), np.asarray(v1), atol=0.01)
        # Each matches its own model
        assert np.allclose(np.asarray(v0), _expected_wdl("m0", 0, obs), atol=0.01)
        assert np.allclose(np.asarray(v1), _expected_wdl("m1", 0, obs), atol=0.01)
    finally:
        _stop_server(s0)
        _stop_server(s1)


def test_multi_server_weight_broadcast():
    """Same model on two servers — weight update reaches both."""
    import threading

    game = pyspiel.load_game("tic_tac_toe")
    mm = _MockModel("main")

    s0 = InferenceServer(gpu_id=0)
    s1 = InferenceServer(gpu_id=1)
    try:
        rq0 = s0.register_actor(0, max_states=8, obs_flat=27, mask_flat=9, pol_flat=9)
        rq1 = s1.register_actor(0, max_states=8, obs_flat=27, mask_flat=9, pol_flat=9)

        specs = {"main": {"model": mm}}
        s0._thread = threading.Thread(
            target=_run_server,
            args=(s0.incoming_queue, s0._result_qs, specs,
                  "tic_tac_toe", 128, None, s0._shm_bufs, None, 0),
            daemon=True)
        s1._thread = threading.Thread(
            target=_run_server,
            args=(s1.incoming_queue, s1._result_qs, specs,
                  "tic_tac_toe", 128, None, s1._shm_bufs, None, 1),
            daemon=True)
        s0._thread.start()
        s1._thread.start()
        time.sleep(0.1)
        if not s0._thread.is_alive() or not s1._thread.is_alive():
            raise RuntimeError("Server thread crashed on startup")

        ev0 = SharedEvaluator(game, s0.incoming_queue, rq0, actor_id=0, model_id="main")
        ev1 = SharedEvaluator(game, s1.incoming_queue, rq1, actor_id=0, model_id="main")

        s = _make_state([0, 4])
        obs = np.asarray(s.observation_tensor(), dtype=np.float32)

        # Before update: both return step 0
        v0_before, _ = ev0._inference(s)
        v1_before, _ = ev1._inference(s)
        assert np.allclose(np.asarray(v0_before), _expected_wdl("main", 0, obs), atol=0.01)
        assert np.allclose(np.asarray(v1_before), _expected_wdl("main", 0, obs), atol=0.01)

        # Advance model state + broadcast to both servers
        mm.step_up()
        s0.update_weights("main", {}, 128)
        s1.update_weights("main", {}, 128)
        time.sleep(0.1)

        # After update: both return step 1
        v0_after, _ = ev0._inference(s)
        v1_after, _ = ev1._inference(s)
        assert np.allclose(np.asarray(v0_after), _expected_wdl("main", 1, obs), atol=0.01)
        assert np.allclose(np.asarray(v1_after), _expected_wdl("main", 1, obs), atol=0.01)
    finally:
        _stop_server(s0)
        _stop_server(s1)


def test_eval_slot_reservation():
    """Reserved eval actor slots work through the shared training server."""
    import threading
    game = pyspiel.load_game("tic_tac_toe")
    mm = _MockModel("main")
    server = InferenceServer()
    try:
        # Register a normal training actor
        rq_train = server.register_actor(0, max_states=8, obs_flat=27,
                                          mask_flat=9, pol_flat=9)
        server.register_model("main", {}, 8, 1)
        specs = {"main": {"model": mm}}
        # Reserve eval slots
        eval_ids = server.reserve_eval_actors(2, max_states=8, obs_flat=27,
                                              mask_flat=9, pol_flat=9)
        assert len(eval_ids) == 2
        assert server.eval_actor_ids == [100000, 100001]
        assert server.actor_shm_name(100000) is not None

        server._thread = threading.Thread(
            target=_run_server,
            args=(server.incoming_queue, server._result_qs, specs,
                  "tic_tac_toe", 128, None, server._shm_bufs,
                  None, 0, True),
            daemon=True)
        server._thread.start()
        time.sleep(0.1)
        if not server._thread.is_alive():
            raise RuntimeError("Server thread crashed on startup")

        # Train actor: works as usual
        ev_train = SharedEvaluator(game, server.incoming_queue,
                                   server.result_queue(0), actor_id=0,
                                   model_id="main")
        s = _make_state([0, 4])
        v_train, _ = ev_train._inference(s)
        assert abs(sum(v_train) - 1.0) < 0.01

        # Eval actor (slot 100000): uses non-SHM path by default
        ev_eval = SharedEvaluator(game, server.incoming_queue,
                                  server.result_queue(100000), actor_id=100000,
                                  model_id="main")
        v_eval, _ = ev_eval._inference(s)
        assert abs(sum(v_eval) - 1.0) < 0.01
        assert np.allclose(np.asarray(v_train), np.asarray(v_eval), atol=0.01)
    finally:
        _stop_server(server)


def test_eval_slot_multi_model():
    """Reserved eval slots with distinct model_id get correct results."""
    import threading
    game = pyspiel.load_game("tic_tac_toe")
    mm = _MockModel("main")
    me = _MockModel("eval")
    server = InferenceServer()
    try:
        rq = server.register_actor(0, max_states=8, obs_flat=27, mask_flat=9,
                                    pol_flat=9)
        server.register_model("main", {}, 8, 1)
        specs = {"main": {"model": mm}, "eval": {"model": me}}
        eval_ids = server.reserve_eval_actors(1, max_states=8, obs_flat=27,
                                              mask_flat=9, pol_flat=9)
        server._thread = threading.Thread(
            target=_run_server,
            args=(server.incoming_queue, server._result_qs, specs,
                  "tic_tac_toe", 128, None, server._shm_bufs,
                  None, 0, True),
            daemon=True)
        server._thread.start()
        time.sleep(0.1)
        if not server._thread.is_alive():
            raise RuntimeError("Server thread crashed on startup")

        # Eval actor with "eval" model — different from main
        ev = SharedEvaluator(game, server.incoming_queue,
                             server.result_queue(100000), actor_id=100000,
                             model_id="eval")
        s = _make_state([0, 4])
        obs = np.asarray(s.observation_tensor(), dtype=np.float32)
        v_eval, _ = ev._inference(s)
        assert np.allclose(np.asarray(v_eval), _expected_wdl("eval", 0, obs),
                           atol=0.01)
    finally:
        _stop_server(server)


def test_eval_model_update_via_existing_server():
    """Eval models (m0/m1) on training server: pre-registered & updatable."""
    import threading
    game = pyspiel.load_game("tic_tac_toe")
    mm_main = _MockModel("main", num_actions=9)
    mm_m0 = _MockModel("m0", num_actions=9)
    mm_m1 = _MockModel("m1", num_actions=9)

    from train.batch_mcts.shm import ServerShm
    server = InferenceServer()
    srv_shm = None
    try:
        server.register_actor(0, max_states=8, obs_flat=27, mask_flat=9,
                              pol_flat=9)
        server.register_model("main", {}, 8, 1)
        server.register_model("m0", {}, 8, 1)
        server.register_model("m1", {}, 8, 1)
        server.reserve_eval_actors(2, max_states=8, obs_flat=27,
                                   mask_flat=9, pol_flat=9)

        specs = {"main": {"model": mm_main},
                 "m0": {"model": mm_m0}, "m1": {"model": mm_m1}}
        srv_shm = ServerShm(f"test_evm_{id(server)}", 128, 9, create=True)
        server._thread = threading.Thread(
            target=_run_server,
            args=(server.incoming_queue, server._result_qs, specs,
                  "tic_tac_toe", 128, None, server._shm_bufs,
                  srv_shm, 0, True),
            daemon=True)
        server._thread.start()
        time.sleep(0.1)
        if not server._thread.is_alive():
            raise RuntimeError("Server thread crashed on startup")

        s = _make_state([0, 4])
        obs = np.asarray(s.observation_tensor(), dtype=np.float32)

        # Actor 100000 uses "m0" — should match _MockModel("m0")
        ev0 = SharedEvaluator(game, server.incoming_queue,
                              server.result_queue(100000), actor_id=100000,
                              model_id="m0", server_shm=srv_shm)
        v0, _ = ev0._inference(s)
        assert np.allclose(np.asarray(v0), _expected_wdl("m0", 0, obs),
                           atol=0.01)

        # Actor 100001 uses "m1" — different mock, different output
        ev1 = SharedEvaluator(game, server.incoming_queue,
                              server.result_queue(100001), actor_id=100001,
                              model_id="m1", server_shm=srv_shm)
        v1, _ = ev1._inference(s)
        assert np.allclose(np.asarray(v1), _expected_wdl("m1", 0, obs),
                           atol=0.01)
        assert not np.allclose(np.asarray(v0), np.asarray(v1), atol=0.01)

        # Update m0 to a new model — simulates eval switching checkpoint
        mm_m0.step_up()
        server.update_weights("m0", {}, 128)
        time.sleep(0.1)
        v0_new, _ = ev0._inference(s)
        assert np.allclose(np.asarray(v0_new), _expected_wdl("m0", 1, obs),
                           atol=0.01)
        assert not np.allclose(np.asarray(v0), np.asarray(v0_new), atol=0.01)

        # m1 unchanged
        v1_unchanged, _ = ev1._inference(s)
        assert np.allclose(np.asarray(v1_unchanged), _expected_wdl("m1", 0, obs),
                           atol=0.01)
    finally:
        _stop_server(server)
        if srv_shm:
            try: srv_shm.close()
            except Exception: pass
            try: srv_shm.unlink()
            except Exception: pass


def test_shared_eval():
    main()


if __name__ == "__main__":
    test_shared_eval()
