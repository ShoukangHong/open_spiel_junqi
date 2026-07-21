"""Tests for XiangqiSymmetry: action mapping, obs transform, value invariance."""

import numpy as np
import pyspiel

from train.model.xiangqi_symmetry import XiangqiSymmetry, COLS, ROWS, NUM_ACTIONS


# ── Helpers ──────────────────────────────────────────────────────────────────

def _legal_mask(state):
    return np.asarray(state.legal_actions_mask(), dtype=bool)


def _obs(state):
    return np.asarray(state.observation_tensor(), dtype=np.float32)


# ── Action mapping ───────────────────────────────────────────────────────────

def test_inv_is_bijection():
    """Each transform's inv is a permutation (no collisions, all 8100 mapped)."""
    sym = XiangqiSymmetry()
    for k in range(4):
        mapping = sym._inv[k]
        assert len(set(mapping)) == NUM_ACTIONS, \
            f"k={k}: not a permutation ({len(set(mapping))} unique)"


def test_identity_is_noop():
    """k=0 is identity: inv[0, a] == a."""
    sym = XiangqiSymmetry()
    for a in range(NUM_ACTIONS):
        assert sym._inv[0, a] == a


def test_mirror_roundtrip():
    """Mirror (k=1) applied twice returns identity."""
    sym = XiangqiSymmetry()
    inv = sym._inv[1]
    twice = inv[inv]
    for a in range(NUM_ACTIONS):
        assert twice[a] == a


def test_swap_roundtrip():
    """Swap (k=2) applied twice returns identity."""
    sym = XiangqiSymmetry()
    inv = sym._inv[2]
    twice = inv[inv]
    for a in range(NUM_ACTIONS):
        assert twice[a] == a


def test_composition_roundtrip():
    """Mirror+Swap (k=3) applied twice returns identity."""
    sym = XiangqiSymmetry()
    inv = sym._inv[3]
    twice = inv[inv]
    for a in range(NUM_ACTIONS):
        assert twice[a] == a


# ── Mask/Permutation correctness ────────────────────────────────────────────

def test_mask_maps_legal_to_legal():
    """After transform, permuted mask corresponds to legal actions in the
    equivalent transformed state."""
    sym = XiangqiSymmetry()
    game = pyspiel.load_game("xiangqi")

    for trial in range(100):
        state = game.new_initial_state()
        for __ in range(8):
            legal = state.legal_actions()
            if not legal:
                break
            state.apply_action(np.random.choice(legal))

        if state.is_terminal():
            continue
        mask = _legal_mask(state)

        for k in range(4):
            inv = sym._inv[k]
            new_mask = mask[inv]
            if new_mask.sum() != mask.sum():
                print(f"FAIL trial={trial} k={k}: {new_mask.sum()} vs {mask.sum()}")
                # Check permutation integrity
                mapped = set(int(inv[i]) for i in range(NUM_ACTIONS))
                unmapped = set(range(NUM_ACTIONS)) - mapped
                dup = len(inv) - len(set(inv))
                print(f"  inv range: {min(inv)}..{max(inv)}  dup={dup}  missing={len(unmapped)}")
                print(f"  mask_len={len(mask)}  inv_len={len(inv)}")
                raise AssertionError(
                    f"trial={trial} k={k}: {new_mask.sum()} vs {mask.sum()}")
    print(f"  all 100 trials passed")


# ── Value invariance ────────────────────────────────────────────────────────

def test_value_is_invariant():
    """WDL is unchanged by all symmetry transforms."""
    sym = XiangqiSymmetry()
    B = 4
    obs = np.random.randn(B, 1530).astype(np.float32)
    mask = np.random.rand(B, NUM_ACTIONS).astype(np.float32) > 0.5
    policy = np.random.rand(B, NUM_ACTIONS).astype(np.float32)
    policy /= policy.sum(axis=-1, keepdims=True)
    value = np.array([[0.8, 0.1, 0.1], [0.3, 0.5, 0.2],
                      [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32)

    _, _, _, new_val = sym.augment_batch(obs, mask, policy, value)
    np.testing.assert_array_equal(new_val, value)


# ── Player-to-move correctness ───────────────────────────────────────────────

def test_swap_flips_player_to_move():
    """Swap (k>=2) inverts player-to-move indicator. Tested on manual apply."""
    # Build obs: Red to move, one Red piece at (6,0), one Black piece at (3,0)
    obs = np.zeros((17, ROWS, COLS), dtype=np.float32)
    obs[0, 6, 0] = 1.0    # Red General at (6,0) — plane 0
    obs[7, 3, 0] = 1.0    # Black General at (3,0) — plane 7
    obs[14, :, :] = 1.0    # Red to move

    # k=2 swap manually: flip rows + swap piece planes + invert turn
    o = obs.copy()
    o = o[:, ::-1, :]                        # rows: 6→3, 3→6
    tmp = o[0:7].copy()
    o[0:7] = o[7:14]
    o[7:14] = tmp                              # swap Red↔Black planes
    o[14] = 1.0 - o[14]                      # invert turn: 1.0→0.0

    assert o[14, 0, 0] == 0.0                # now Black to move
    assert o[7, 3, 0] > 0.5                  # Red piece moved to (3,0) (was Black at (3,0) originally)
    assert o[0, 6, 0] > 0.5                  # Black piece moved to (6,0)

    # Swap twice → identity
    o2 = o.copy()
    o2 = o2[:, ::-1, :]
    tmp2 = o2[0:7].copy()
    o2[0:7] = o2[7:14]
    o2[7:14] = tmp2
    o2[14] = 1.0 - o2[14]
    np.testing.assert_allclose(o2, obs, atol=0.01)


# ── Visualisation ────────────────────────────────────────────────────────────

_RED = [None, "帥", "仕", "相", "傌", "俥", "炮", "兵"]   # ptype 1-7
_BLK = [None, "將", "士", "象", "馬", "車", "砲", "卒"]
_EMPTY = "＋"  # full-width intersection marker

_CCOLS = ["一", "二", "三", "四", "五", "六", "七", "八", "九"]   # cols 0-8
_CROWS = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]  # rows 0-9

K_NAMES = ["Identity", "Mirror(L-R)", "Swap(color+V)", "Mirror+Swap"]


def _obs_to_board_str(obs):
    """Convert flat [1530] obs to a 10×9 Chinese character grid."""
    obs = obs.reshape(17, ROWS, COLS)
    rows = []
    header = "    " + "".join(_CCOLS)
    rows.append(header)
    for r in range(ROWS):
        line = f"{_CROWS[r]:>2s} "
        for c in range(COLS):
            ch = _EMPTY
            for ptype in range(1, 8):
                if obs[ptype - 1, r, c] > 0.5:
                    ch = _RED[ptype]
                    break
                if obs[ptype + 6, r, c] > 0.5:
                    ch = _BLK[ptype]
                    break
            line += ch
        rows.append(line)
    rows.append(header)
    return "\n".join(rows)


def _action_str(a):
    """Format action as (from_row,from_col)->(to_row,to_col)."""
    fs, ts = a // 90, a % 90
    return f"({fs//COLS},{fs%COLS})→({ts//COLS},{ts%COLS})"


def _top_moves_str(policy, mask, inv, n=8):
    """Show transformed actions with original action, weight, and legality."""
    legal = np.where(mask)[0]
    # Sort by policy value descending
    ranked = sorted(range(len(policy)), key=lambda a: -policy[a])
    out = ""
    shown = 0
    for a in ranked:
        if policy[a] < 0.001:
            break
        orig_a = inv[a]
        is_legal = "✓" if mask[a] else "✗ ILLEGAL"
        out += (f"  {_action_str(a):>20s}  "
                f"p={policy[a]:.4f}  {is_legal}"
                f"  ← orig={_action_str(orig_a)}\n")
        shown += 1
        if shown >= n:
            break
    return out


def test_write_symmetry_viz():
    """Write all 4 symmetry transforms to temp/symmetry_viz.txt."""
    import os
    game = pyspiel.load_game("xiangqi")
    state = game.new_initial_state()
    # Play a few random legal moves for a non-trivial position
    rng = np.random.RandomState(42)
    for _ in range(6):
        legal = state.legal_actions()
        if not legal:
            break
        state.apply_action(int(rng.choice(legal)))

    obs = _obs(state)
    mask = _legal_mask(state)

    # Non-uniform policy: assign high weights to 3 specific legal moves
    # so we can track whether the transform correctly remaps them.
    legal = np.where(mask)[0]
    policy = np.zeros(NUM_ACTIONS, dtype=np.float32)
    noise = np.random.RandomState(7).uniform(0.001, 0.005, size=NUM_ACTIONS)
    policy[legal] = noise[legal]
    # Pick 3 identifiable legal moves and assign high weights
    picks = [legal[0], legal[len(legal)//3], legal[2*len(legal)//3]]
    weights = [0.50, 0.25, 0.12]
    for a, w in zip(picks, weights):
        policy[a] = w
    policy /= policy.sum()
    value = np.array([0.7, 0.2, 0.1], dtype=np.float32)

    sym = XiangqiSymmetry()

    import tempfile as _tf
    out_path = os.path.join(_tf.gettempdir(), "symmetry_viz.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        # Show original state with Chinese glyphs
        f.write("Original state:\n")
        f.write(_obs_to_board_str(obs))
        player0 = "Red" if obs.reshape(17, ROWS, COLS)[14, 0, 0] > 0.5 else "Black"
        f.write(f"\n\nPlayer: {player0}    Value: W={value[0]:.2f} D={value[1]:.2f} L={value[2]:.2f}\n")
        f.write("\nTop moves (original):\n")
        f.write(_top_moves_str(policy, mask, np.arange(NUM_ACTIONS)))

        for k in range(4):
            f.write(f"\n{'=' * 60}\n")
            f.write(f"  Transform {k}: {K_NAMES[k]}\n")
            f.write(f"{'=' * 60}\n\n")

            o = obs.reshape(17, ROWS, COLS).copy()
            mirror = k % 2 == 1
            swap = k >= 2
            if mirror:
                o = o[:, :, ::-1]
            if swap:
                o = o[:, ::-1, :]
                tmp = o[0:7].copy()
                o[0:7] = o[7:14]
                o[7:14] = tmp
                o[14] = 1.0 - o[14]

            player = "Red" if o[14, 0, 0] > 0.5 else "Black"
            f.write(f"Player to move: {player}\n\n")
            f.write(_obs_to_board_str(o.flatten()))
            f.write(f"\n\nValue: W={value[0]:.2f} D={value[1]:.2f} L={value[2]:.2f}\n")

            inv = sym._inv[k]
            pm = mask[inv]
            pp = policy[inv]
            illegal_count = int(((pp > 0.001) & ~pm).sum())
            f.write("\nTop moves (transformed policy):\n")
            f.write(_top_moves_str(pp, pm, inv))
            if illegal_count > 0:
                f.write(f"  *** WARNING: {illegal_count} illegal actions with non-zero policy!\n")

    print(f"[symmetry_viz] {out_path}")


# ── Move-number randomisation ─────────────────────────────────────────────────

def test_move_num_randomised_when_below_threshold():
    """move_num < MOVE_NUM_RAND_MAX → re-rolled to another integer below it."""
    sym = XiangqiSymmetry()
    obs = np.zeros((1, 1530), dtype=np.float32)
    o = obs.reshape(17, 10, 9)
    count = 50
    o[15] = count / sym.MAX_GAME_LENGTH
    mask = np.ones((1, NUM_ACTIONS), dtype=bool)
    policy = np.ones((1, NUM_ACTIONS), dtype=np.float32) / NUM_ACTIONS

    found_changed = False
    for _ in range(50):
        new_obs, _, _, _ = sym.augment_batch(obs.copy(), mask.copy(), policy.copy())
        o2 = new_obs.reshape(17, 10, 9)
        new_mn = float(o2[15, 0, 0]) * sym.MAX_GAME_LENGTH
        assert new_mn < sym.MOVE_NUM_RAND_MAX, \
            f"move_num should stay < {sym.MOVE_NUM_RAND_MAX}: {new_mn}"
        assert abs(round(new_mn) - new_mn) < 0.01, "should be integer"
        if abs(new_mn - count) > 0.1:
            found_changed = True
    assert found_changed, "move_num should change at least once in 50 trials"


def test_move_num_unchanged_when_above_threshold():
    """move_num >= MOVE_NUM_RAND_MAX → never modified."""
    sym = XiangqiSymmetry()
    obs = np.zeros((1, 1530), dtype=np.float32)
    o = obs.reshape(17, 10, 9)
    count = sym.MOVE_NUM_RAND_MAX + 20
    o[15] = count / sym.MAX_GAME_LENGTH
    mask = np.ones((1, NUM_ACTIONS), dtype=bool)
    policy = np.ones((1, NUM_ACTIONS), dtype=np.float32) / NUM_ACTIONS

    for _ in range(20):
        new_obs, _, _, _ = sym.augment_batch(obs.copy(), mask.copy(), policy.copy())
        o2 = new_obs.reshape(17, 10, 9)
        assert abs(float(o2[15, 0, 0]) * sym.MAX_GAME_LENGTH - count) < 0.01, \
            "move_num >= threshold should not be randomised"


# ── No-capture randomisation ─────────────────────────────────────────────────

def test_no_cap_reduced_for_decisive_game():
    """draw < 0.3 AND no_cap < threshold → counter decreases, integer steps."""
    sym = XiangqiSymmetry()
    obs = np.zeros((1, 1530), dtype=np.float32)
    o = obs.reshape(17, 10, 9)
    count = 5
    o[16] = count / sym.MAX_NO_CAP_LENGTH
    mask = np.ones((1, NUM_ACTIONS), dtype=bool)
    policy = np.ones((1, NUM_ACTIONS), dtype=np.float32) / NUM_ACTIONS
    value = np.array([[0.8, 0.1, 0.1]], dtype=np.float32)  # decisive

    found_reduced = False
    for _ in range(200):
        np.random.seed(None)
        new_obs, _, _, _ = sym.augment_batch(
            obs.copy(), mask.copy(), policy.copy(), value.copy())
        o2 = new_obs.reshape(17, 10, 9)
        new_nc = float(o2[16, 0, 0]) * sym.MAX_NO_CAP_LENGTH
        assert abs(round(new_nc) - new_nc) < 0.01, \
            f"no_cap should be integer: {new_nc}"
        assert new_nc <= count + 0.01, \
            f"no_cap should not increase: {new_nc:.1f} > {count}"
        if new_nc < count - 0.1:
            found_reduced = True

    assert found_reduced, "no_cap should decrease at least once in 200 trials"


def test_no_cap_untouched_for_draw_game():
    """draw >= 0.3 → no_cap never modified."""
    sym = XiangqiSymmetry()
    obs = np.zeros((1, 1530), dtype=np.float32)
    o = obs.reshape(17, 10, 9)
    count = 5
    o[16] = count / sym.MAX_NO_CAP_LENGTH
    mask = np.ones((1, NUM_ACTIONS), dtype=bool)
    policy = np.ones((1, NUM_ACTIONS), dtype=np.float32) / NUM_ACTIONS
    value = np.array([[0.2, 0.6, 0.2]], dtype=np.float32)  # draw-heavy

    for _ in range(30):
        np.random.seed(None)
        new_obs, _, _, _ = sym.augment_batch(
            obs.copy(), mask.copy(), policy.copy(), value.copy())
        o2 = new_obs.reshape(17, 10, 9)
        assert abs(float(o2[16, 0, 0]) * sym.MAX_NO_CAP_LENGTH - count) < 0.01, \
            f"no_cap should not change for draw game: {float(o2[16, 0, 0]):.4f}"


def test_no_cap_untouched_above_threshold():
    """no_cap >= threshold → never modified even for decisive games."""
    sym = XiangqiSymmetry()
    obs = np.zeros((1, 1530), dtype=np.float32)
    o = obs.reshape(17, 10, 9)
    # Set at the threshold boundary: orig_count == NO_CAP_RAND_MAX
    # is NOT < NO_CAP_RAND_MAX, so it should not be touched.
    count = sym.NO_CAP_RAND_MAX
    o[16] = count / sym.MAX_NO_CAP_LENGTH
    mask = np.ones((1, NUM_ACTIONS), dtype=bool)
    policy = np.ones((1, NUM_ACTIONS), dtype=np.float32) / NUM_ACTIONS
    value = np.array([[0.9, 0.05, 0.05]], dtype=np.float32)

    for _ in range(20):
        np.random.seed(None)
        new_obs, _, _, _ = sym.augment_batch(
            obs.copy(), mask.copy(), policy.copy(), value.copy())
        o2 = new_obs.reshape(17, 10, 9)
        new_nc = float(o2[16, 0, 0]) * sym.MAX_NO_CAP_LENGTH
        assert abs(new_nc - count) < 0.01, \
            f"no_cap >= threshold should not change: {new_nc:.4f} vs {count}"
