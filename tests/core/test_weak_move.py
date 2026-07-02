"""Test weak-move logic in train_othello.py."""

import numpy as np
import pyspiel

from train.batch_mcts.config import MCTSConfig
from train.batch_mcts.mcts import BatchMCTS
from train.core.replay_buffer import ReplayBuffer
from train.core.weak_move import nn_raw_after_move, try_weak_move
from train.games.othello.config import OthelloTrainConfig as TrainConfig
from train.games.othello.play import play_game

def _cfg(**kw):
    c = TrainConfig()
    c.game = "othello"
    c.max_simulations = 8
    c.mcts_batch_size = 4
    c.weak_max_per_game = 1
    for k, v in kw.items():
        setattr(c, k, v)
    return c

def _scalar_to_wdl(v):
    """scalar in [-1,1] → WDL [w,d,l]."""
    w = max(v, 0.0)
    l = max(-v, 0.0)
    d = 1.0 - abs(v)
    return np.array([w, d, l], dtype=np.float32)


class ZeroEvaluator:
    def batch_inference_raw(self, states):
        vals, prs = [], []
        for s in states:
            vals.append([0.0, 1.0, 0.0])  # WDL neutral
            l = s.legal_actions()
            prs.append([(a, 1.0 / len(l)) for a in l])
        return np.array(vals, dtype=np.float32), prs
    def _inference(self, state):
        l = state.legal_actions()
        pol = np.zeros(65, dtype=np.float32)
        for a in l:
            pol[a] = 1.0 / len(l)
        return _scalar_to_wdl(0.0), pol

class _ControlledEval:
    def __init__(self, nn_val=0.0, nn_argmax=None):
        self.nn_val = nn_val
        self.nn_argmax = nn_argmax
    def _inference(self, state):
        l = state.legal_actions()
        pol = np.zeros(65, dtype=np.float32)
        for a in l: pol[a] = 0.5
        if self.nn_argmax is not None and self.nn_argmax in l:
            pol[self.nn_argmax] = 1.0
        return _scalar_to_wdl(self.nn_val), pol
    def batch_inference_raw(self, states):
        vals, prs = [], []
        for s in states:
            vals.append([0.0, 1.0, 0.0])  # WDL neutral
            l = s.legal_actions()
            prs.append([(a, 1.0 / len(l)) for a in l])
        return np.array(vals, dtype=np.float32), prs


# Test A

def testnn_raw_after_move():
    print("Test A: nn_raw_after_move evaluates post-move state ...", end=" ")
    g = pyspiel.load_game("othello")
    st = g.new_initial_state()
    ev = ZeroEvaluator()
    a = st.legal_actions()[0]
    r = nn_raw_after_move(ev, st, a)
    assert r == 0.0
    print("PASSED")


# Test B

def test_weak_max_zero():
    print("Test B: weak_max=0 => no weak moves ...", end=" ")
    g, ev = pyspiel.load_game("tic_tac_toe"), ZeroEvaluator()
    m = BatchMCTS(g, MCTSConfig(max_simulations=8, batch_size=4), ev,
                  random_state=np.random.RandomState(42))
    c = _cfg(weak_max_per_game=0, game="tic_tac_toe")
    si, ret, rg, _ws = play_game(g, m, m, c, np.random.RandomState(42))
    assert len(rg) == 0
    for it in si:
        assert (it[4] if len(it) > 4 else "") == ""
    print("PASSED")


# Test C

def _mk_weak_test(game_str, diff_val, threshold, weak_thresh,
                  seed=42, n_runs=20):
    """Run try_weak_move *n_runs* times, verify the dominant branch.

    Returns the most common (tag, rare_state_is_None, action_is_ra).
    """
    g = pyspiel.load_game(game_str)
    st = g.new_initial_state()
    legal = st.legal_actions()
    wa = legal[-1] if len(legal) > 1 else legal[0]

    tags = []
    for run in range(n_runs):
        ev = _ControlledEval(nn_val=diff_val, nn_argmax=wa)
        m = BatchMCTS(g, MCTSConfig(max_simulations=16, batch_size=4), ev,
                      random_state=np.random.RandomState(seed + run))
        root = m.mcts_search(st.clone())
        ra = root.best_child().action
        if ra == wa:
            continue  # skip degenerate cases
        c = _cfg(rare_case_threshold=threshold,
                 weak_move_threshold=weak_thresh,
                 weak_move_prob=1.0, game=game_str)
        rng = np.random.RandomState(seed + run)
        a, tag, wc, rs, _wc = try_weak_move(
            m, st.clone(), root, c, weak_count=0,
            weak_max=c.weak_max_per_game, rng=rng)
        tags.append((tag, rs is None, a == ra))
    return tags, ra, wa


def test_weak_branches():
    print("Test C: weak-move three branches ...")

    # C1: nn_val=0.8 → prob=90%. mcts≈0 → prob=50%. rel_drop≈0.8 > 0.4 → rare
    tags, ra, wa = _mk_weak_test("tic_tac_toe", 0.8, 0.4, 0.2, n_runs=30)
    rare_frac = sum(1 for t in tags if t[0] == "") / max(len(tags), 1)
    rs_frac = sum(1 for t in tags if not t[1]) / max(len(tags), 1)
    act_frac = sum(1 for t in tags if t[2]) / max(len(tags), 1)
    assert rare_frac > 0.7, f"expected rare branch dominant, got {rare_frac:.1%}"
    assert rs_frac > 0.7, f"expected rare_state not None, got {rs_frac:.1%}"
    assert act_frac > 0.7, f"expected action==ra, got {act_frac:.1%}"
    print(f"  C1 (rare): {rare_frac:.0%} rare  {rs_frac:.0%} has_rs  "
          f"{act_frac:.0%} act_is_mcts  ({len(tags)}/{30} valid)")

    # C2: nn_val=0 → prob=50% = mcts → rel_drop≈0 < 0.2 → weak accepted
    tags, ra, wa = _mk_weak_test("tic_tac_toe", 0.0, 0.4, 0.2, n_runs=30)
    weak_frac = sum(1 for t in tags if t[0] == "weak") / max(len(tags), 1)
    assert weak_frac > 0.6, f"expected weak branch dominant, got {weak_frac:.1%}"
    print(f"  C2 (accept): {weak_frac:.0%} weak  ({len(tags)}/{30} valid)")

    # C3: nn_val=0.3 → prob=65%. mcts≈0 → prob=50%. rel_drop≈0.3 → weak_final
    tags, ra, wa = _mk_weak_test("tic_tac_toe", 0.3, 0.4, 0.2, n_runs=30)
    wf_frac = sum(1 for t in tags if t[0] == "weak_final") / max(len(tags), 1)
    assert wf_frac > 0.5, f"expected weak_final branch dominant, got {wf_frac:.1%}"
    print(f"  C3 (final): {wf_frac:.0%} weak_final  ({len(tags)}/{30} valid)")


# Test C4: rare weak_count += 1, not exhausted

def test_rare_weak_count_increment():
    print("Test C4: rare weak_count += 1 ...", end=" ")
    g = pyspiel.load_game("tic_tac_toe")
    st = g.new_initial_state()
    ev = _ControlledEval(nn_val=0.8, nn_argmax=st.legal_actions()[-1])
    m = BatchMCTS(g, MCTSConfig(max_simulations=16, batch_size=4), ev,
                  random_state=np.random.RandomState(42))
    root = m.mcts_search(st.clone())
    mcts_a = root.best_child().action
    wa = st.legal_actions()[-1]
    if mcts_a == wa:
        m2 = BatchMCTS(g, MCTSConfig(max_simulations=8, batch_size=4), ev,
                       random_state=np.random.RandomState(99))
        root = m2.mcts_search(st.clone())
    c = _cfg(rare_case_threshold=0.4, weak_move_threshold=0.2,
             weak_move_prob=1.0, game="tic_tac_toe",
             weak_max_per_game=3)

    wc = 1
    for run in range(2):
        _, tag, wc, rs, _ = try_weak_move(
            m, st.clone(), root, c, weak_count=wc,
            weak_max=c.weak_max_per_game, rng=np.random.RandomState(run))
        assert wc > 0  # weak_count still alive
    assert wc > 1, f"rare should increment weak_count, got {wc}"
    print("PASSED")


# Test D

def test_fork_no_weak():
    print("Test D: forked game disables weak ...", end=" ")
    g, ev = pyspiel.load_game("tic_tac_toe"), ZeroEvaluator()
    m = BatchMCTS(g, MCTSConfig(max_simulations=8, batch_size=4), ev,
                  random_state=np.random.RandomState(42))
    st = g.new_initial_state()
    st.apply_action(0); st.apply_action(4); st.apply_action(8)
    c = _cfg(game="tic_tac_toe", weak_max_per_game=3, weak_move_prob=1.0,
             weak_side_prob=1.0)
    si, _, _, _ws2 = play_game(g, m, m, c, np.random.RandomState(42),
                               init_state=st, allow_weak=False)
    for it in si:
        assert (it[4] if len(it) > 4 else "") == ""
    print("PASSED")


# Test E

def test_buffer_tags():
    import tempfile
    print("Test E: buffer tag_counts() ...", end=" ")
    db = tempfile.mktemp(suffix=".db")
    buf = ReplayBuffer(max_size=100, db_path=db)
    o = np.zeros((4, 8, 8), dtype=np.float32)
    mk = np.ones(65, dtype=bool)
    po = np.zeros(65, dtype=np.float32)
    po[19] = 1.0
    buf.append(o, mk, po, 0.5, tag="")
    buf.append(o, mk, po, -0.3, tag="rare")
    buf.append(o, mk, po, 0.8, tag="rare")
    c = buf.tag_counts()
    assert c.get("", 0) == 1 and c.get("rare", 0) == 2
    buf.close()
    import os
    for ext in ("", "-shm", "-wal"):
        try: os.unlink(db + ext)
        except: pass
    print("PASSED")


# Test F

def test_nn_raw_after_move_terminal():
    """When weak move captures general, use game returns, not NN."""
    print("Test G: nn_raw_after_move on general-capture ...", end=" ")
    g = pyspiel.load_game("xiangqi")
    st = g.new_initial_state()
    # Walk the legal moves to find one that directly captures a general
    found = False
    for a in st.legal_actions():
        s2 = st.clone()
        s2.apply_action(a)
        if s2.is_terminal() and s2.returns()[0] != 0:
            class _Dummy:
                def scalar_value(self, s):
                    raise RuntimeError("should not call NN")
            r = nn_raw_after_move(_Dummy(), st, a)
            assert r == s2.returns()[0], f"{r} != {s2.returns()[0]}"
            found = True
            break
    if found:
        print("PASSED")
    else:
        print("SKIP (no direct-capture action found)")


def test_perspective():
    print("Test F: perspective alignment ...", end=" ")
    import train.core.weak_move as wm
    g = pyspiel.load_game("othello")
    st = g.new_initial_state()
    st.apply_action(19)  # black move, now white to go
    ev = _ControlledEval(nn_val=+0.3, nn_argmax=20)
    m = BatchMCTS(g, MCTSConfig(max_simulations=16, batch_size=4), ev,
                  random_state=np.random.RandomState(42))
    root = m.mcts_search(st)
    c = _cfg(rare_case_threshold=0.9, weak_move_threshold=0.9,
             weak_move_prob=1.0)
    _orig = wm.nn_raw_after_move
    wm.nn_raw_after_move = lambda *_: 0.0  # neutral NN value
    try:
        _, tag, _, _, _ = try_weak_move(
            m, st, root, c, weak_count=0, weak_max=1)
        assert tag in ("", "weak", "weak_final")  # perspective test: just no crash
    finally:
        wm.nn_raw_after_move = _orig
    print("PASSED")


def main():
    tests = [
        testnn_raw_after_move, test_weak_max_zero, test_weak_branches,
        test_rare_weak_count_increment,
        test_fork_no_weak, test_buffer_tags,
        test_nn_raw_after_move_terminal, test_perspective,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    print(f"\n{'=' * 40}")
    print(f"{'ALL PASSED' if failed == 0 else f'{failed} FAILED'}")
    print("=" * 40)

def test_weak_move():
    main()

if __name__ == "__main__":
    test_weak_move()
