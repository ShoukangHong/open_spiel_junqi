"""Integration tests: shm server vs direct model, plus concurrent correctness."""

import time
import threading
import numpy as np
import pyspiel

from train.batch_mcts.shm import ActorShm, ServerShm
from train.batch_mcts.shared_evaluator import (
    InferenceServer, SharedEvaluator, _run_server)
from train.batch_mcts.evaluator import PyTorchEvaluator
from train.model.othello_resnet import OthelloResNet
from train.model.model import Model


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_model():
    net = OthelloResNet(input_channels=4, output_size=65, nn_width=8, nn_depth=1)
    return Model(net, device="cpu")


def _make_states(n=3):
    game = pyspiel.load_game("othello")
    states = []
    for _ in range(n):
        s = game.new_initial_state()
        for __ in range(np.random.randint(0, 8)):
            legal = s.legal_actions()
            if not legal:
                break
            s.apply_action(np.random.choice(legal))
        states.append(s)
    return states


# ── Correctness: shm server output matches direct model ──────────────────────

def test_shm_vs_direct_model():
    game = pyspiel.load_game("othello")
    model = _make_model()
    states = _make_states(5)

    # Get direct results
    direct_vals = []
    direct_pols = []
    for s in states:
        obs = np.asarray(s.observation_tensor(), dtype=np.float32)
        mask = np.asarray(s.legal_actions_mask(), dtype=bool)
        v2, pol2 = model.inference(obs, mask)
        direct_vals.append(v2)
        direct_pols.append(pol2)

    # Start server + actor with shm
    obs_flat = int(np.prod(game.observation_tensor_shape()))
    act_flat = game.num_distinct_actions()

    server = InferenceServer()
    try:
        rq = server.register_actor(0, max_states=8, obs_flat=obs_flat,
                                   mask_flat=act_flat, pol_flat=act_flat)
        server.register_model("main", model._model.state_dict(), 8, 1)
        # Create ServerShm for the test (normally done in server.start())
        out_name = f"test_osh_{id(server)}"
        srv_shm = ServerShm(out_name, 128, act_flat, create=True)
        shm_name = server.actor_shm_name(0)
        server._thread = threading.Thread(
            target=_run_server,
            args=(server.incoming_queue, server._result_qs,
                  server._model_specs, "othello", 128, None,
                  server._shm_bufs, srv_shm),
            daemon=True)
        server._thread.start()
        time.sleep(0.1)
        if not server._thread.is_alive():
            raise RuntimeError("Server thread crashed on startup")

        actor_shm = ActorShm(shm_name, 8, obs_flat, act_flat, create=False)
        ev = SharedEvaluator(game, server.incoming_queue, rq, actor_id=0,
                             model_id="main", actor_shm=actor_shm,
                             server_shm=srv_shm)

        vals_shm, priors_shm = ev.batch_inference_raw(states)

        for i, s in enumerate(states):
            v_direct = direct_vals[i]
            p_direct = direct_pols[i]
            v_shm = vals_shm[i]
            prior = priors_shm[i]
            p_shm = np.zeros(act_flat, dtype=np.float32)
            for a, pr in prior:
                p_shm[a] = pr

            np.testing.assert_allclose(v_direct, v_shm, atol=1e-5,
                                       err_msg=f"value mismatch state {i}")
            np.testing.assert_allclose(p_direct, p_shm, atol=1e-5,
                                       err_msg=f"policy mismatch state {i}")
    finally:
        _stop_server(server)
        _safe_cleanup_shm(srv_shm)


# ── Concurrency ──────────────────────────────────────────────────────────────

def _stop_server(server):
    try:
        server.stop()
    except Exception:
        pass
    try:
        server._thread.join(timeout=3)
    except Exception:
        pass


def _safe_cleanup_shm(shm):
    import gc
    gc.collect()
    try:
        shm.close()
    except Exception:
        pass
    try:
        shm.unlink()
    except Exception:
        pass


def test_shm_concurrent():
    game = pyspiel.load_game("othello")
    model = _make_model()
    states = _make_states(20)

    obs_flat = int(np.prod(game.observation_tensor_shape()))
    act_flat = game.num_distinct_actions()

    server = InferenceServer()
    srv_shm = None
    try:
        n_actors = 3
        evs = []
        for i in range(n_actors):
            rq = server.register_actor(i, max_states=8, obs_flat=obs_flat,
                                       mask_flat=act_flat, pol_flat=act_flat)
            rq  # keep reference
        server.register_model("main", model._model.state_dict(), 8, 1)
        out_name = f"test_csh_{id(server)}"
        srv_shm = ServerShm(out_name, 128, act_flat, create=True)
        server._thread = threading.Thread(
            target=_run_server,
            args=(server.incoming_queue, server._result_qs,
                  server._model_specs, "othello", 128, None,
                  server._shm_bufs, srv_shm),
            daemon=True)
        server._thread.start()
        time.sleep(0.1)
        if not server._thread.is_alive():
            raise RuntimeError("Server thread crashed on startup")
        time.sleep(0.05)

        for i in range(n_actors):
            shm_name = server.actor_shm_name(i)
            shm = ActorShm(shm_name, 8, obs_flat, act_flat, create=False)
            ev = SharedEvaluator(game, server.incoming_queue,
                                 server.result_queue(i), actor_id=i,
                                 model_id="main", actor_shm=shm,
                                 server_shm=srv_shm)
            evs.append(ev)

        errors = []
        lock = threading.Lock()

        def worker(ev, sidx):
            s = states[sidx]
            obs = np.asarray(s.observation_tensor(), dtype=np.float32)
            mask = np.asarray(s.legal_actions_mask(), dtype=bool)
            v_direct, p_direct = model.inference(obs, mask)
            try:
                v_shm, _ = ev._inference(s)
                v_shm = np.asarray(v_shm)
                if not np.allclose(v_direct, v_shm, atol=1e-5):
                    with lock:
                        errors.append(f"worker {sidx}: value mismatch")
            except Exception as e:
                with lock:
                    errors.append(f"worker {sidx}: {type(e).__name__}: {e}")

        threads = []
        for i in range(20):
            t = threading.Thread(target=worker, args=(evs[i % n_actors], i))
            threads.append(t)
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"{len(errors)} errors: {errors[:3]}"
    finally:
        _stop_server(server)
        if srv_shm:
            _safe_cleanup_shm(srv_shm)
