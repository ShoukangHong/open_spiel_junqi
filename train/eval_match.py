"""Evaluate two strategies against each other in Othello.

Usage:  python eval_match.py

Also importable:  from train.eval_match import run_match
"""

import logging
import json
import queue
import sys
import os
_sys_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _sys_root not in sys.path:
    sys.path.insert(0, _sys_root)

import numpy as np
import pyspiel

from train.batch_mcts.config import MCTSConfig
from train.batch_mcts.evaluator import PyTorchEvaluator
from train.batch_mcts.mcts import BatchMCTS
from train.batch_mcts.shared_evaluator import fast_legal_mask
from train.model.othello_resnet import Model, OthelloResNet

# ── Module-level caches (reused across run_match calls) ──────────────────────
_game = None
_models = {}
_mcts_bots = {}
_mcts_evaluators = {}
_opening_book = None
_opening_prob = 0.0


def _get_build_fn(game_name):
    if game_name == "xiangqi":
        from train.core.model_builder import build_xiangqi_model
        return build_xiangqi_model
    from train.core.model_builder import build_othello_model
    return build_othello_model


def _game_obj(game_name=None):
    global _game
    if _game is None or game_name is not None:
        _game = pyspiel.load_game(game_name or "othello")
    return _game


def _model_for(player_cfg):
    key = (player_cfg["checkpoint_dir"], player_cfg.get("checkpoint_step", 0))
    if key not in _models:
        config_path = os.path.join(player_cfg["checkpoint_dir"], "train_config.json")
        with open(config_path) as f:
            tc = json.load(f)
        tc["path"] = player_cfg["checkpoint_dir"]  # override saved config
        build_fn = _get_build_fn(tc["game"])
        game = pyspiel.load_game(tc["game"])
        m = build_fn(game, tc)
        step = player_cfg.get("checkpoint_step", 0)
        if step > 0:
            m.load_checkpoint(step)
            logging.info(f"  [eval] loaded {player_cfg['checkpoint_dir'].rsplit(chr(92),1)[-1].rsplit('/',1)[-1]}:{step}"
                         f"  params={m.num_trainable_variables}")
        _models[key] = m
    return _models[key]


def _mcts_for(player_cfg):
    key = (player_cfg["checkpoint_dir"], player_cfg.get("checkpoint_step", 0))
    if key not in _mcts_bots:
        # Read game name from train_config.json
        config_path = os.path.join(player_cfg["checkpoint_dir"], "train_config.json")
        with open(config_path) as f:
            tc = json.load(f)
        game = pyspiel.load_game(tc["game"])
        model = _model_for(player_cfg)
        ev = PyTorchEvaluator(game, model)
        _mcts_evaluators[key] = ev
        cfg = MCTSConfig(
            max_simulations=player_cfg.get("mcts_simulations", 128),
            batch_size=player_cfg.get("mcts_batch_size", 4),
            uct_c=player_cfg.get("mcts_uct_c", 1.41),
            draw_penalty=tc.get("draw_penalty", 0.0),
            repeat_penalty=tc.get("repeat_penalty", 0.1),
            fpu_lambda=tc.get("fpu_lambda", 0.2),
            probe_depth=tc.get("probe_depth", 0),
            probe_surprise=tc.get("probe_surprise", 0.3),
            policy_epsilon=0, verbose=False)
        _mcts_bots[key] = BatchMCTS(
            game, cfg, ev, random_state=np.random.RandomState(42))
    return _mcts_bots[key]


def _act(player_cfg, state, move_num, temperature, temp_drop):
    strategy = player_cfg["strategy"]
    legal = state.legal_actions()

    if strategy == "random":
        return np.random.choice(legal)

    if strategy == "greedy":
        cur = state.current_player()
        token = "x" if cur == 0 else "o"
        counts = [str(state.clone().apply_action(a)).count(token)
                  for a in legal]
        return legal[int(np.argmax(counts))]

    if strategy == "model":
        obs = np.asarray(state.observation_tensor(), dtype=np.float32)
        mask = fast_legal_mask(state, state.get_game().num_distinct_actions())
        _, policy = _model_for(player_cfg).inference(obs, mask)
        probs = np.array([policy[a] for a in legal])
        tau = 2/3 if move_num < temp_drop else temperature
        if tau > 0:
            probs = probs ** (1.0 / max(tau, 0.01))
            probs /= probs.sum()
            return np.random.choice(legal, p=probs)
        return legal[int(np.argmax(probs))]

    if strategy == "mcts":
        from train.core.policy import select_action_with_adv
        from train.batch_mcts.mcts import compute_solved_policy
        root = _mcts_for(player_cfg).mcts_search(state)
        solved = compute_solved_policy(
            root.children, state.current_player(),
            state.get_game().max_utility(),
            root_visits=root.explore_count)
        tau = 2/3 if move_num < temp_drop else temperature
        a, _ = select_action_with_adv(root, state, alpha=0.0,
                                      temperature=tau, base_policy=solved)
        return a

    raise ValueError(f"Unknown strategy: {strategy}")


# ── Core function ────────────────────────────────────────────────────────────

def run_match(cfg0, cfg1, num_games=100, temperature=0.1, temp_drop=5,
              quiet=True):
    """Run a match between two strategies.

    Args:
        cfg0, cfg1: dicts with keys: strategy, checkpoint_dir, checkpoint_step,
                    mcts_simulations, mcts_batch_size, mcts_uct_c.
        num_games: number of games to play.
        temperature: model/MCTS temperature after temp_drop.
        temp_drop: first N moves use τ=1 for MCTS.
        quiet: if True, suppress per-player load messages.

    Returns:
        (score_dict, sequences)
        score_dict: {"name0": wins, "name1": wins, "draw": draws}
        sequences: list of (label, tuple_of_actions)
    """
    s0, s1 = cfg0["strategy"], cfg1["strategy"]
    st0 = cfg0.get("checkpoint_step", 0)
    st1 = cfg1.get("checkpoint_step", 0)
    name0 = (f"{s0}" if s0 in ("random", "greedy")
             else f"{s0}(step{st0},{cfg0.get('mcts_simulations', 0)}sim)")
    name1 = (f"{s1}" if s1 in ("random", "greedy")
             else f"{s1}(step{st1},{cfg1.get('mcts_simulations', 0)}sim)")

    # Pre-load models and determine game
    game_name = "othello"
    for cfg in (cfg0, cfg1):
        s = cfg["strategy"]
        if s in ("model", "mcts"):
            _model_for(cfg)
            config_path = os.path.join(cfg["checkpoint_dir"], "train_config.json")
            with open(config_path) as f:
                game_name = json.load(f).get("game", "othello")
    game = pyspiel.load_game(game_name)

    if not quiet:
        print(f"\nMatch: {name0} (black) vs {name1} (white), {num_games} games\n")

    score = {name0: 0, name1: 0, "draw": 0}
    sequences = []

    # Load opening book once
    global _opening_book, _opening_prob
    if _opening_book is None:
        tc_path = os.path.join(cfg0.get("checkpoint_dir", ""), "train_config.json")
        if os.path.exists(tc_path):
            with open(tc_path) as f:
                tc = json.load(f)
        else:
            tc = {}
        _opening_prob = 1.0  # always use opening book for eval
        opening_dir = tc.get("opening_book_dir", "")
        if opening_dir and _opening_prob > 0:
            from train.games.opening_book import OpeningBook
            _opening_book = OpeningBook(game, opening_dir)

    for i in range(num_games):
        if i % 2 == 0:
            cfg_b, cfg_w = cfg0, cfg1
            label_b, label_w = name0, name1
        else:
            cfg_b, cfg_w = cfg1, cfg0
            label_b, label_w = name1, name0

        if _opening_book and np.random.random() < _opening_prob:
            state = _opening_book.sample(np.random)
        else:
            state = game.new_initial_state()
        move_num = 0
        moves = []
        while not state.is_terminal():
            cur = state.current_player()
            cfg = cfg_b if cur == 0 else cfg_w
            action = _act(cfg, state, move_num, temperature, temp_drop)
            state.apply_action(action)
            moves.append(action)
            move_num += 1
        sequences.append((label_b, tuple(moves)))
        r = state.returns()[0]

        if r > 0:
            score[label_b] += 1
        elif r < 0:
            score[label_w] += 1
        else:
            score["draw"] += 1

        if not quiet and (i + 1) % 10 == 0:
            print(f"  {i + 1:4d}/{num_games}  "
                  f"{name0}={score[name0]:3d}  {name1}={score[name1]:3d}  "
                  f"draw={score['draw']:3d}")

    return score, sequences


# ── Parallel version ─────────────────────────────────────────────────────────

def run_match_parallel(cfg0, cfg1, num_games=100, temperature=0.1,
                       temp_drop=5, num_actors=4, quiet=True,
                       inference_batch_size=128, existing_server=None):
    """Parallel eval using shared GPU inference server.

    Spawns N game processes sharing one GPU process for batched inference.
    Uses multiprocessing.Process to bypass GIL on MCTS search.

    If *existing_server* is an InferenceServer, eval actors connect to it
    instead of spawning a new GPU process.  Pre-reserved actor slots and
    models are registered via weight-update messages.
    """
    import multiprocessing as mp
    from train.batch_mcts.shared_evaluator import (
        InferenceServer, SharedEvaluator)
    import json, os

    s0, s1 = cfg0["strategy"], cfg1["strategy"]
    st0 = cfg0.get("checkpoint_step", 0)
    st1 = cfg1.get("checkpoint_step", 0)
    name0 = (f"{s0}" if s0 in ("random", "greedy")
             else f"{s0}(step{st0},{cfg0.get('mcts_simulations', 0)}sim)")
    name1 = (f"{s1}" if s1 in ("random", "greedy")
             else f"{s1}(step{st1},{cfg1.get('mcts_simulations', 0)}sim)")

    # Read game name from the first NN-based player config
    game_name = "othello"  # fallback
    tc = {}
    for _, pcfg in [("m0", cfg0), ("m1", cfg1)]:
        if pcfg["strategy"] in ("mcts", "model"):
            config_path = os.path.join(pcfg["checkpoint_dir"],
                                        "train_config.json")
            with open(config_path) as f:
                tc = json.load(f)
            game_name = tc.get("game", "othello")
            break
    _game_obj(game_name)  # ensure cached game matches the eval game

    if existing_server is not None:
        # Reuse training server — let server load checkpoints from disk
        srv = existing_server
        for mid, pcfg in [("m0", cfg0), ("m1", cfg1)]:
            step = pcfg.get("checkpoint_step", 0)
            if pcfg["strategy"] in ("mcts", "model") and step > 0:
                ckpt_path = os.path.join(pcfg["checkpoint_dir"],
                                         f"checkpoint-{step}.pt")
                srv._incoming.put((mid, "load", ckpt_path,
                                   inference_batch_size))
        actor_ids = srv.eval_actor_ids
        if len(actor_ids) < num_actors:
            raise RuntimeError(
                f"Need {num_actors} eval slots, got {len(actor_ids)}. "
                f"Increase eval_num_actors in config.")
        actor_ids = actor_ids[:num_actors]
        actor_rqs = [srv.result_queue(aid) for aid in actor_ids]
        shm_names = [srv.actor_shm_name(aid) for aid in actor_ids]
        srv_shm_name = srv.server_shm_name
    else:
        srv = InferenceServer(build_model_fn=_get_build_fn(game_name),
                              pin_cpu=False)
        for mid, pcfg in [("m0", cfg0), ("m1", cfg1)]:
            if pcfg["strategy"] not in ("mcts", "model"):
                continue
            model_objs = _model_for(pcfg)
            srv.register_model(mid, model_objs._model.state_dict(),
                               tc.get("nn_width", 32), tc.get("nn_depth", 6))
        actor_rqs = [srv.register_actor(i) for i in range(num_actors)]
        actor_ids = list(range(num_actors))
        shm_names = [srv.actor_shm_name(i) for i in range(num_actors)]
        srv_shm_name = srv.server_shm_name
        srv.start(game_name, inference_batch_size)

    score = {name0: 0, name1: 0, "draw": 0}
    sequences = []

    if not quiet:
        print(f"\nMatch: {name0} (black) vs {name1} (white), "
              f"{num_games} games, {num_actors} actors\n")

    # Spawn actor processes
    stats_q = mp.Queue()
    procs = []
    games_per = num_games // num_actors
    for w in range(num_actors):
        gs = w * games_per
        ge = num_games if w == num_actors - 1 else gs + games_per
        p = mp.Process(target=_eval_actor_process, args=(
            actor_ids[w], gs, ge, cfg0, cfg1, name0, name1, temperature,
            temp_drop, srv.incoming_queue, actor_rqs[w], stats_q, game_name,
            shm_names[w], srv_shm_name))
        p.start()
        procs.append(p)

    # Collect results
    for _ in range(num_actors):
        local_score, local_seqs = stats_q.get()
        for k in score:
            score[k] += local_score[k]
        sequences.extend(local_seqs)

    for p in procs:
        p.join(timeout=5)

    for p in procs:
        if p.is_alive():
            p.terminate()

    # Drain stale results from eval actor queues (prevent cross-iteration
    # contamination when the same actor_id/result_q is reused).
    for rq in actor_rqs:
        while True:
            try:
                rq.get_nowait()
            except queue.Empty:
                break

    if existing_server is None:
        srv.stop()
        srv.shutdown()
    return score, sequences


def _eval_actor_process(worker_id, g_start, g_end, cfg0, cfg1,
                        name0, name1, temperature, temp_drop,
                        incoming_q, result_q, stats_out, game_name,
                        shm_name=None, server_shm_name=None):
    """Actor process body — runs at module level for Windows spawn compatibility."""
    import numpy as np
    import pyspiel
    from train.batch_mcts.shared_evaluator import SharedEvaluator

    game = pyspiel.load_game(game_name)

    actor_shm = None
    server_shm = None
    if shm_name is not None and server_shm_name is not None:
        from train.batch_mcts.shm import ActorShm, ServerShm
        obs_flat = int(np.prod(game.observation_tensor_shape()))
        act_flat = game.num_distinct_actions()
        mcts_bs = cfg0.get('mcts_batch_size', 8) or cfg1.get('mcts_batch_size', 8)
        actor_shm = ActorShm(shm_name, mcts_bs, obs_flat, act_flat, create=False)
        server_shm = ServerShm(server_shm_name, 128, act_flat, create=False)

    evals = {}
    for mid, pcfg in [("m0", cfg0), ("m1", cfg1)]:
        if pcfg["strategy"] in ("mcts", "model"):
            evals[mid] = SharedEvaluator(
                game, incoming_q, result_q, actor_id=worker_id, model_id=mid,
                actor_shm=actor_shm, server_shm=server_shm)
        else:
            evals[mid] = None

    # Opening book (load once per actor)
    obook = None
    obook_prob = 0.0
    tc_path = os.path.join(cfg0.get("checkpoint_dir", ""), "train_config.json")
    if os.path.exists(tc_path):
        with open(tc_path) as f:
            _tc = json.load(f)
        obook_prob = 1.0  # always use opening book for eval
        obook_dir = _tc.get("opening_book_dir", "")
        if obook_dir and obook_prob > 0:
            from train.games.opening_book import OpeningBook
            obook = OpeningBook(game, obook_dir)

    rng = np.random.RandomState(worker_id * 1000 + g_start)
    local_score = {name0: 0, name1: 0, "draw": 0}
    local_seqs = []

    for i in range(g_start, g_end):
        if rng.random() < 0.5:
            cfg_b, cfg_w = cfg0, cfg1
            label_b, label_w = name0, name1
            ev_b, ev_w = evals["m0"], evals["m1"]
        else:
            cfg_b, cfg_w = cfg1, cfg0
            label_b, label_w = name1, name0
            ev_b, ev_w = evals["m1"], evals["m0"]

        if obook and rng.random() < obook_prob:
            state = obook.sample(rng)
        else:
            state = game.new_initial_state()
        init_state = state.clone()  # for replay (may differ from initial state)
        move_num = 0
        moves = []
        while not state.is_terminal():
            cur = state.current_player()
            cfg = cfg_b if cur == 0 else cfg_w
            ev = ev_b if cur == 0 else ev_w
            action = _act_parallel(cfg, state, move_num,
                                    temperature, temp_drop, ev,
                                    worker_id)
            state.apply_action(action)
            moves.append(action)
            move_num += 1
        local_seqs.append((label_b, tuple(moves), init_state))
        r = state.returns()[0]
        if r > 0:
            local_score[label_b] += 1
        elif r < 0:
            local_score[label_w] += 1
        else:
            local_score["draw"] += 1

    stats_out.put((local_score, local_seqs))


def _act_parallel(player_cfg, state, move_num, temperature, temp_drop,
                  shared_eval=None, worker_id=0):
    strategy = player_cfg["strategy"]
    legal = state.legal_actions()

    if strategy == "random":
        return np.random.choice(legal)

    if strategy == "greedy":
        cur = state.current_player()
        token = "x" if cur == 0 else "o"
        counts = [str(state.clone().apply_action(a)).count(token)
                  for a in legal]
        return legal[int(np.argmax(counts))]

    if strategy == "model":
        _, policy = shared_eval._inference(state)
        probs = np.array([policy[a] for a in legal], dtype=np.float64)
        probs = np.nan_to_num(probs, nan=0.0).clip(min=0)
        s = probs.sum()
        if s <= 0:
            return np.random.choice(legal)
        if temperature > 0:
            probs = probs ** (1.0 / max(temperature, 0.01))
            probs /= probs.sum()
            return np.random.choice(legal, p=probs)
        return np.random.choice(legal, p=probs / s)

    if strategy == "mcts":
        if not hasattr(_act_parallel, '_mcts_cache'):
            _act_parallel._mcts_cache = {}
        key = (player_cfg["checkpoint_dir"],
               player_cfg.get("checkpoint_step", 0),
               worker_id)
        if key not in _act_parallel._mcts_cache:
            import json, os
            config_path = os.path.join(player_cfg["checkpoint_dir"],
                                       "train_config.json")
            with open(config_path) as f:
                tc = json.load(f)
            cfg = MCTSConfig(
                max_simulations=player_cfg.get("mcts_simulations", 128),
                batch_size=player_cfg.get("mcts_batch_size", 4),
                uct_c=player_cfg.get("mcts_uct_c", 1.41),
                draw_penalty=tc.get("draw_penalty", 0.0),
                repeat_penalty=tc.get("repeat_penalty", 0.1),
                fpu_lambda=tc.get("fpu_lambda", 0.2),
                probe_depth=tc.get("probe_depth", 0),
                probe_surprise=tc.get("probe_surprise", 0.3),
                policy_epsilon=0, verbose=False)
            _act_parallel._mcts_cache[key] = BatchMCTS(
                pyspiel.load_game(tc["game"]), cfg, shared_eval,
                random_state=np.random.RandomState(worker_id * 1000))
        mcts = _act_parallel._mcts_cache[key]
        from train.core.policy import select_action_with_adv
        from train.batch_mcts.mcts import compute_solved_policy
        root = mcts.mcts_search(state)
        solved = compute_solved_policy(
            root.children, state.current_player(),
            state.get_game().max_utility(),
            root_visits=root.explore_count)
        tau = 2/3 if move_num < temp_drop else temperature
        a, _ = select_action_with_adv(root, state, alpha=0.0,
                                      temperature=tau, base_policy=solved)
        return a

    raise ValueError(f"Unknown strategy: {strategy}")



def main():
    score, sequences = run_match_parallel(
        PLAYER[0], PLAYER[1],
        num_games=DEFAULT_NUM_GAMES,
        temperature=DEFAULT_TEMPERATURE,
        temp_drop=DEFAULT_TEMP_DROP,
        num_actors=20,
        quiet=False)

    n = DEFAULT_NUM_GAMES
    s0, s1 = list(score.keys())[:2]
    print(f"\n-- {s0}: {score[s0]} ({score[s0]/n:.1%})  "
          f"{s1}: {score[s1]} ({score[s1]/n:.1%})  "
          f"draw: {score['draw']} ({score['draw']/n:.1%})")

    # Move diversity
    print("\n[PREFIX OVERLAP]  unique prefixes / games  (most-common count)")
    max_len = max(len(item[1]) for item in sequences)
    for L in range(1, min(max_len + 1, 61), 5):
        prefs = [item[1][:L] for item in sequences if len(item[1]) >= L]
        if not prefs:
            continue
        arr = np.array(prefs, dtype=int)
        unique, counts = np.unique(arr, axis=0, return_counts=True)
        top_count = counts.max()
        pct = top_count / len(prefs) * 100
        print(f"  first {L:>2d} moves:  {len(unique)} unique  "
              f"(top occurs {top_count}/{len(prefs)} = {pct:.0f}%)")

    # ── Save detailed game log ───────────────────────────────────────
    import datetime
    out_dir = os.path.join(_sys_root, "temp")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    log_path = os.path.join(out_dir, f"eval_{ts}.log")
    with open(log_path, "w", encoding="utf-8") as f:
        frame_names = [k for k in score if k != "draw"]
        f.write(f"# {frame_names[0]} vs {frame_names[1]}\n")
        f.write(f"# Score: {score}\n\n")
        _write_eval_games(f, sequences, score)
    print(f"[eval] Log saved to {log_path}")

def _write_eval_games(f, sequences, score):
    """Replay game sequences and write detailed board-by-board logs."""
    game = _game_obj()
    n = len(sequences)
    frame_names = [k for k in score if k != "draw"]
    name0, name1 = frame_names[0], frame_names[1]
    for gi, item in enumerate(sequences):
        label, moves = item[0], item[1]
        init_state = item[2] if len(item) > 2 else None
        if init_state is not None:
            state = init_state.clone()
        else:
            state = game.new_initial_state()
        f.write(f"\n{'=' * 60}\nGame {gi + 1}/{n}  label={label}\n{'=' * 60}\n")
        for mi, a in enumerate(moves):
            cur = state.current_player()
            pname = "BLACK(p0)" if cur == 0 else "WHITE(p1)"
            f.write(f"\n── Move {mi + 1}  {pname}  action={a}"
                    f" ({state.action_to_string(cur, a)}) ──\n")
            state.apply_action(a)
            f.write(f"{state}\n")
        r = state.returns()
        if r[0] > 0:
            winner = f"{label} (Black)"
        elif r[0] < 0:
            winner = f"{name1 if label == name0 else name0} (White)"
        else:
            winner = "draw"
        f.write(f"\n── Final (move {len(moves)}) ──\n"
                f"Result: {winner} wins  Returns: {r[0]:+.0f}/{r[1]:+.0f}\n")


DEFAULT_NUM_GAMES = 40
DEFAULT_TEMPERATURE = 0.03
DEFAULT_TEMP_DROP = 5

PLAYER = {
    0: {"strategy": "mcts",
        "checkpoint_dir": r"C:\Users\shouk\othello_train\cloud_wdl_mix",
        "checkpoint_step": 540,
        "mcts_simulations": 512, "mcts_batch_size": 8, "mcts_uct_c": 1.41},
    # 1: {"strategy": "mcts", # 早期的benchmark
    #     "checkpoint_dir": r"C:\Users\shouk\othello_train\cloud_wdl_argmax", # argmax 240 us benchmark
    #     "checkpoint_step": 240,
    #     "mcts_simulations": 128, "mcts_batch_size": 8, "mcts_uct_c": 1.41},
    # 1: {"strategy": "mcts", # 中期的benchmark
    #     "checkpoint_dir": r"C:\Users\shouk\othello_train\cloud_wdl_b",
    #     "checkpoint_step": 990,
    #     "mcts_simulations": 128, "mcts_batch_size": 8, "mcts_uct_c": 1.41},
    1: {"strategy": "mcts", # 晚期的benchmark，很强了
        "checkpoint_dir": r"C:\Users\shouk\othello_train\cloud_wdl_128",
        "checkpoint_step": 1500,
        "mcts_simulations": 512, "mcts_batch_size": 8, "mcts_uct_c": 1.41},
}

PLAYER = {
    0: {"strategy": "mcts",
        "checkpoint_dir": r"C:\Users\shouk\xiangqi_train\cloud_re",
        "checkpoint_step": 500,
        "mcts_simulations": 600, "mcts_batch_size": 10, "mcts_uct_c": 4.0},
    # 1: {"strategy": "mcts", # 早期的benchmark
    #     "checkpoint_dir": r"C:\Users\shouk\othello_train\cloud_wdl_argmax", # argmax 240 us benchmark
    #     "checkpoint_step": 240,
    #     "mcts_simulations": 128, "mcts_batch_size": 8, "mcts_uct_c": 1.41},
    # 1: {"strategy": "mcts", # 中期的benchmark
    #     "checkpoint_dir": r"C:\Users\shouk\othello_train\cloud_wdl_b",
    #     "checkpoint_step": 990,
    #     "mcts_simulations": 128, "mcts_batch_size": 8, "mcts_uct_c": 1.41},
    1: {"strategy": "mcts",
        "checkpoint_dir": r"C:\Users\shouk\xiangqi_train\cloud_re",
        "checkpoint_step": 390,
        "mcts_simulations": 600, "mcts_batch_size": 10, "mcts_uct_c": 4.0},
}

if __name__ == "__main__":
    import multiprocessing as _mp
    _mp.set_start_method("spawn", force=True)
    main()
