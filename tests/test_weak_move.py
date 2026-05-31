"""Test weak-move logic in train_othello.py."""

import numpy as np
import pyspiel

from train.batch_mcts.config import MCTSConfig
from train.batch_mcts.mcts import BatchMCTS
from train.train_othello import (
    _nn_raw_after_move, _try_weak_move, play_game, ReplayBuffer, TrainConfig
)

def _cfg(**kw):
    c = TrainConfig()
    c.game = "othello"
    c.max_simulations = 8
    c.mcts_batch_size = 4
    c.weak_max_per_game = 1
    for k, v in kw.items():
        setattr(c, k, v)
    return c

class ZeroEvaluator:
    def batch_inference_raw(self, states):
        vals, prs = [], []
        for s in states:
            vals.append(0.0)
            l = s.legal_actions()
            prs.append([(a, 1.0 / len(l)) for a in l])
        return np.array(vals), prs
    def _inference(self, state):
        l = state.legal_actions()
        pol = np.zeros(65, dtype=np.float32)
        for a in l:
            pol[a] = 1.0 / len(l)
        return 0.0, pol

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
        return self.nn_val, pol
    def batch_inference_raw(self, states):
        vals, prs = [], []
        for s in states:
            vals.append(0.0)
            l = s.legal_actions()
            prs.append([(a, 1.0 / len(l)) for a in l])
        return np.array(vals), prs


# Test A

def test_nn_raw_after_move():
    print("Test A: _nn_raw_after_move evaluates post-move state ...", end=" ")
    g = pyspiel.load_game("othello")
    st = g.new_initial_state()
    ev = ZeroEvaluator()
    a = st.legal_actions()[0]
    r = _nn_raw_after_move(ev, st, a)
    assert r == 0.0
    print("PASSED")


# Test B

def test_weak_max_zero():
    print("Test B: weak_max=0 => no weak moves ...", end=" ")
    g, ev = pyspiel.load_game("tic_tac_toe"), ZeroEvaluator()
    m = BatchMCTS(g, MCTSConfig(max_simulations=8, batch_size=4), ev,
                  random_state=np.random.RandomState(42))
    c = _cfg(weak_max_per_game=0, game="tic_tac_toe")
    si, ret, rg, _ws = play_game(g, m, c, np.random.RandomState(42))
    assert len(rg) == 0
    for it in si:
        assert (it[4] if len(it) > 4 else "") == ""
    print("PASSED")


# Test C

def _mk_weak_test(game_str, diff_val, threshold, weak_thresh):
    g = pyspiel.load_game(game_str)
    st = g.new_initial_state()
    legal = st.legal_actions()

    # Pick a weak action that differs from what MCTS would likely choose.
    # Use the LAST legal action as weak pick (MCTS tends to pick earlier ones).
    wa = legal[-1] if len(legal) > 1 else legal[0]

    ev = _ControlledEval(nn_val=diff_val, nn_argmax=wa)
    m = BatchMCTS(g, MCTSConfig(max_simulations=16, batch_size=4), ev,
                  random_state=np.random.RandomState(42))
    root = m.mcts_search(st)
    ra = root.best_child().action
    mv = root.total_reward / max(root.explore_count, 1)

    # Guard: if MCTS happens to pick the same action, the weak-move
    # logic short-circuits (weak_a == mcts_action) and our threshold
    # test is meaningless.
    if ra == wa:
        # Re-seed MCTS to get a different best action
        m2 = BatchMCTS(g, MCTSConfig(max_simulations=8, batch_size=4), ev,
                       random_state=np.random.RandomState(99))
        root = m2.mcts_search(st)
        ra = root.best_child().action
    assert ra != wa, f"MCTS chose weak_act={wa} — can't test threshold"

    c = _cfg(rare_case_threshold=threshold, weak_move_threshold=weak_thresh,
             weak_move_prob=1.0, game=game_str)
    a, tag, wc, rs, _wc = _try_weak_move(
        m, st, root, c, weak_count=0, weak_max=c.weak_max_per_game)
    return a, tag, wc, rs, ra, wa, mv


def test_weak_branches():
    print("Test C: weak-move three branches ...")

    # C1: nn_val=0.8 → prob=90%. mcts≈0 → prob=50%. rel_drop≈0.8 > 0.4 → rare
    a, tag, wc, rs, ra, wa, mv = _mk_weak_test("tic_tac_toe", 0.8, 0.4, 0.2)
    assert tag == "rare"
    assert rs is not None
    assert a == ra
    print("  C1 (rare): PASSED")

    # C2: nn_val=0 → prob=50% = mcts → rel_drop≈0 < 0.2 → weak accepted
    a, tag, wc, rs, ra, wa, mv = _mk_weak_test("tic_tac_toe", 0.0, 0.4, 0.2)
    assert tag == ""
    assert rs is None
    assert a != ra  # weak_a picked, not MCTS action
    print("  C2 (accept): PASSED")

    # C3: nn_val=0.3 → prob=65%. mcts≈0 → prob=50%. rel_drop≈0.3 → weak_final
    a, tag, wc, rs, ra, wa, mv = _mk_weak_test("tic_tac_toe", 0.3, 0.4, 0.2)
    assert tag == ""
    assert rs is None
    assert a != ra  # weak_a picked, not MCTS action
    assert wc > 1
    print("  C3 (final): PASSED")


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
    si, _, _, _ws2 = play_game(g, m, c, np.random.RandomState(42),
                               init_state=st, allow_weak=False)
    for it in si:
        assert (it[4] if len(it) > 4 else "") == ""
    print("PASSED")


# Test E

def test_buffer_tags():
    print("Test E: buffer tag_counts() ...", end=" ")
    buf = ReplayBuffer(max_size=100)
    o = np.zeros((4, 8, 8), dtype=np.float32)
    mk = np.ones(65, dtype=bool)
    po = np.zeros(65, dtype=np.float32)
    po[19] = 1.0
    buf.append(o, mk, po, 0.5, tag="")
    buf.append(o, mk, po, -0.3, tag="rare")
    buf.append(o, mk, po, 0.8, tag="rare")
    c = buf.tag_counts()
    assert c.get("", 0) == 1 and c.get("rare", 0) == 2
    print("PASSED")


# Test F

def test_perspective():
    print("Test F: perspective alignment ...", end=" ")
    import train.train_othello as tto
    g = pyspiel.load_game("othello")
    st = g.new_initial_state()
    st.apply_action(19)  # black move, now white to go
    ev = _ControlledEval(nn_val=+0.3, nn_argmax=20)
    m = BatchMCTS(g, MCTSConfig(max_simulations=16, batch_size=4), ev,
                  random_state=np.random.RandomState(42))
    root = m.mcts_search(st)
    c = _cfg(rare_case_threshold=0.4, weak_move_threshold=0.2,
             weak_move_prob=1.0)
    _orig = tto._nn_raw_after_move
    tto._nn_raw_after_move = lambda *_: 0.3
    try:
        _, tag, _, _, _ = _try_weak_move(
            m, st, root, c, weak_count=0, weak_max=1)
        assert tag == ""  # nn_cur=-0.3, mcts≈0, diff≈-0.3 < 0.4
    finally:
        tto._nn_raw_after_move = _orig
    print("PASSED")


def main():
    tests = [
        test_nn_raw_after_move, test_weak_max_zero, test_weak_branches,
        test_fork_no_weak, test_buffer_tags, test_perspective,
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
