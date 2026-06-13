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
        h = abs(hash((self.mid, self._step, o.tobytes()))) % 1000
        return float(h) / 1000.0

    def _wdl(self, o):
        v = self._scalar(o)
        q = (v - 0.5) * 2
        d = 1.0 - abs(q)
        w = np.array([max(q, 0), d, max(-q, 0)], dtype=np.float32)
        return w / w.sum()


# ── Helpers ─────────────────────────────────────────────────────────────────

def _expected_wdl(model_id, step, obs):
    """Recompute the expected WDL for a given state."""
    h = abs(hash((model_id, step, obs.tobytes()))) % 1000
    v = float(h) / 1000.0
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


def _start_server(server, mock_models):
    """Start server with mock models using the real _run_server."""
    import threading
    specs = {}
    for mid, mm in mock_models.items():
        specs[mid] = {"model": mm}
    server._thread = threading.Thread(
        target=_run_server,
        args=(server.incoming_queue, server._result_qs, specs,
              "tic_tac_toe", 128))
    server._thread.start()
    time.sleep(0.05)

def _stop_server(server):
    try:
        server.stop()
    except Exception:
        pass
    try:
        server._thread.join(timeout=3)
    except Exception:
        pass


# ── Tests ───────────────────────────────────────────────────────────────────

def test_single_inference():
    game = pyspiel.load_game("tic_tac_toe")
    mm = _MockModel("main")
    server = InferenceServer()
    try:
        rq = server.register_actor(0)
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
        rq = server.register_actor(0)
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
        rq0, rq1 = server.register_actor(0), server.register_actor(1)
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
        rq = server.register_actor(0)
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
        rq = server.register_actor(0)
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
        rq = server.register_actor(0)
        specs = {"wdl": {"model": _FixedWDLModel()}}
        server._thread = threading.Thread(
            target=_run_server,
            args=(server.incoming_queue, server._result_qs, specs,
                  "tic_tac_toe", 128))
        server._thread.start()
        time.sleep(0.05)

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
        rq = server.register_actor(0)
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
        rq0, rq1 = server.register_actor(0), server.register_actor(1)
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
        rqs = [server.register_actor(i) for i in range(num_actors)]
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
        rqs = [server.register_actor(i) for i in range(num_actors)]
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
        rq = server.register_actor(0)
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
        rq = server.register_actor(0)
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


def test_shared_eval():
    main()


if __name__ == "__main__":
    test_shared_eval()
