"""Integration tests for shared GPU evaluator with multi-model support."""

import time

import numpy as np
import pyspiel

from train.batch_mcts.shared_evaluator import (
    InferenceServer, SharedEvaluator, _run_server)


class _MockModel:
    """Mock model: output = f(model_id, step, state_hash).

    batch_inference returns values and policies matching the production
    format (scalar or WDL depending on value_classes).
    """

    def __init__(self, model_id: str, value_classes: int = 1):
        self.mid = model_id
        self._step = 0
        self._vc = value_classes

    def step_up(self):
        self._step += 1

    def batch_inference(self, obs, mask):
        n = obs.shape[0]
        if self._vc == 1:
            values = np.array([self._scalar(obs[i]) for i in range(n)],
                              dtype=np.float32)
        else:
            values = np.array([self._wdl(obs[i]) for i in range(n)],
                              dtype=np.float32)
        policies = np.zeros((n, 65), dtype=np.float32)
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

def _expected_scalar(model_id, step, obs):
    h = abs(hash((model_id, step, obs.tobytes()))) % 1000
    return float(h) / 1000.0


def _make_state(moves=()):
    game = pyspiel.load_game("tic_tac_toe")
    s = game.new_initial_state()
    for a in moves:
        s.apply_action(a)
    return s


def _start_server(server, mock_models, value_classes=1):
    """Start server with mock models using the real _run_server."""
    import threading
    specs = {}
    for mid, mm in mock_models.items():
        specs[mid] = {
            "model": mm,
            "value_classes": value_classes,
        }
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
        assert abs(float(v) - _expected_scalar("main", 0, obs)) < 0.01
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
            assert abs(float(values[i]) - _expected_scalar("main", 0, obs)) < 0.01
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

        assert abs(float(v0) - _expected_scalar("m0", 0, obs)) < 0.01
        assert abs(float(v1) - _expected_scalar("m1", 0, obs)) < 0.01
        assert abs(float(v0) - float(v1)) > 0.01
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
        assert abs(float(v0) - _expected_scalar("main", 0, obs)) < 0.01

        mm.step_up()
        server._incoming.put(("main", {}, 128))
        time.sleep(0.1)

        v1, _ = ev._inference(s)
        assert abs(float(v1) - _expected_scalar("main", 1, obs)) < 0.01
    finally:
        _stop_server(server)


def test_wdl_mode():
    game = pyspiel.load_game("tic_tac_toe")
    mm = _MockModel("main_wdl", value_classes=3)
    server = InferenceServer()
    try:
        rq = server.register_actor(0)
        _start_server(server, {"main_wdl": mm}, value_classes=3)

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
        def batch_inference(self, obs, mask):
            n = obs.shape[0]
            return (np.array([[0.8, 0.1, 0.1]] * n, dtype=np.float32),
                    np.ones((n, 65), dtype=np.float32) / 65)

    server = InferenceServer()
    try:
        rq = server.register_actor(0)
        specs = {"wdl": {"model": _FixedWDLModel(), "value_classes": 3}}
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
            assert abs(float(v0) - _expected_scalar("m0", 0, obs)) < 0.01
            v1, _ = ev1._inference(s)
            assert abs(float(v1) - _expected_scalar("m1", 0, obs)) < 0.01
            assert abs(float(v0) - float(v1)) > 0.01
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

        assert abs(float(v00) - _expected_scalar("m0", 0, obs)) < 0.01
        assert abs(float(v10) - _expected_scalar("m0", 0, obs)) < 0.01
        assert abs(float(v01) - _expected_scalar("m1", 0, obs)) < 0.01
        assert abs(float(v11) - _expected_scalar("m1", 0, obs)) < 0.01
        assert abs(float(v00) - float(v01)) > 0.01
    finally:
        _stop_server(server)


def _expected_wdl(model_id, step, obs):
    v = _expected_scalar(model_id, step, obs)
    q = (v - 0.5) * 2
    d = 1.0 - abs(q)
    w = np.array([max(q, 0), d, max(-q, 0)], dtype=np.float32)
    return w / w.sum()


def _stress_worker(actor_id, ev0, ev1, game, states, errors, lock,
                   mid0="m0", mid1="m1", wdl=False):
    rng = np.random.RandomState(actor_id * 1000)
    for _ in range(30):
        mid = rng.randint(0, 2)
        ev = ev0 if mid == 0 else ev1
        model_id = mid0 if mid == 0 else mid1
        s = states[rng.randint(0, len(states))]
        obs = np.asarray(s.observation_tensor(), dtype=np.float32)

        value, policy = ev._inference(s)

        if wdl:
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
        else:
            expected_val = _expected_scalar(model_id, 0, obs)
            if abs(float(value) - expected_val) >= 0.01:
                with lock:
                    errors.append(f"[a{actor_id} m{mid}] value: "
                                  f"{float(value):.3f} != {expected_val:.3f}")

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
    m0 = _MockModel("m0_wdl", value_classes=3)
    m1 = _MockModel("m1_wdl", value_classes=3)
    server = InferenceServer()
    try:
        num_actors = 4
        rqs = [server.register_actor(i) for i in range(num_actors)]
        _start_server(server, {"m0_wdl": m0, "m1_wdl": m1}, value_classes=3)

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
                           mid0="m0_wdl", mid1="m1_wdl", wdl=True)

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
