"""Generate opening book positions with Pikafish — score-proximity filtered.

Launches pikafish, recursively expands moves whose score is within ±50 cp of
the best move, up to *depth* plies.  Saves leaf positions as .txt files
(folder-per-depth, compatible with OpeningBook loader).
"""

import os
import subprocess
import sys

import numpy as np
import pyspiel

PIKAFISH = r"C:\Users\shouk\AI\pikafish\pikafish-avx2.exe"
OUT_DIR = r"C:\Users\shouk\xiangqi_train\cloud_new\opening_book"
DEPTH = 3          # plies to expand
SCORE_WINDOW = 38   # cp — moves within this of best are kept
MAX_BRANCH = 10     # cap branches per position (below depth 1)
SEARCH_DEPTH = 10   # engine search depth per position
GAME = pyspiel.load_game("xiangqi")
BUFFER_DB = os.path.join(os.path.dirname(OUT_DIR), "opening_book.db")

# Hash for pikafish (MB)
HASH_MB = 256
THREADS = 4

# Buffer writer (lazy init)
_buffer = None


def _save_to_buffer(state, cp_score):
    """Append position to opening book replay buffer for viewer."""
    global _buffer
    if _buffer is None:
        from train.core.replay_buffer import ReplayBuffer
        _buffer = ReplayBuffer(max_size=10000, db_path=BUFFER_DB)
    obs = np.asarray(state.observation_tensor(), dtype=np.float32)
    mask = np.asarray(state.legal_actions_mask(), dtype=bool)
    # Placeholder policy — uniform over legal actions
    n_legal = max(mask.sum(), 1)
    policy = mask.astype(np.float32) / n_legal
    # Value: score/100 mapped to WDL
    v = np.clip(cp_score / 100.0, -1.0, 1.0)  # cp → [-1, 1]
    wdl = np.array([max(v, 0), 1.0 - abs(v), max(-v, 0)], dtype=np.float32)
    _buffer.append(obs, mask, policy, wdl, tag="opening", step=0)


def _send(proc, cmd):
    proc.stdin.write(cmd + "\n")
    proc.stdin.flush()


def _read_until(proc, marker):
    lines = []
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        lines.append(line.strip())
        if line.strip().startswith(marker):
            break
    return lines


def _action_to_uci(a):
    """pyspiel action → UCI move string (Pikafish row numbering)."""
    fr, fc = divmod(a // 90, 9)
    tr, tc = divmod(a % 90, 9)
    # Pikafish rows are flipped: row 9 in pyspiel = row 0 in Pikafish
    return f"{chr(ord('a') + fc)}{9 - fr}{chr(ord('a') + tc)}{9 - tr}"


def _uci_to_action(uci, state):
    """UCI move string → pyspiel action (Pikafish row numbering)."""
    fc = ord(uci[0]) - ord('a')
    fr = 9 - int(uci[1])
    tc = ord(uci[2]) - ord('a')
    tr = 9 - int(uci[3])
    a = (fr * 9 + fc) * 90 + (tr * 9 + tc)
    if a in state.legal_actions():
        return a
    return None


def _set_position(proc, state):
    """Send position to engine via UCI position fen or startpos + moves."""
    history = state.history()
    if not history:
        _send(proc, "position startpos")
        return
    moves = " ".join(_action_to_uci(a) for a in history)
    _send(proc, f"position startpos moves {moves}")


def get_scored_moves(proc, state, score_window=SCORE_WINDOW, search_depth=SEARCH_DEPTH):
    """Get all legal moves scored by the engine. Returns [(action, cp_score), ...]."""
    _set_position(proc, state)
    n_legal = len(state.legal_actions())
    _send(proc, f"setoption name MultiPV value {n_legal}")
    _send(proc, f"go depth {search_depth}")
    output = _read_until(proc, "bestmove")

    scores = {}
    for line in output:
        if line.startswith("info depth") and "multipv" in line and "score cp" in line:
            parts = line.split()
            try:
                pv_idx = parts.index("multipv")
                multipv = int(parts[pv_idx + 1])
                score_idx = parts.index("cp")
                score = int(parts[score_idx + 1])
                pv_idx2 = parts.index("pv")
                moves_uci = parts[pv_idx2 + 1:]
                if moves_uci:
                    first_move_uci = moves_uci[0]
                    a = _uci_to_action(first_move_uci, state)
                    if a is not None:
                        if a not in scores:
                            scores[a] = (score, multipv)
            except (ValueError, IndexError):
                continue

    # Sort by multipv (best first) and return actions within score window
    result = []
    for a, (score, mpv) in sorted(scores.items(), key=lambda x: x[1][1]):
        if abs(score) <= score_window:
            result.append((a, score, mpv))
    return result


def _save_position(state, best_cp, ply, out_dirs):
    """Save one position to .txt and buffer."""
    depth_dir = out_dirs[ply]
    parts = []
    for a in state.history():
        fr, fc = divmod(a // 90, 9)
        tr, tc = divmod(a % 90, 9)
        parts.append(f"{fr}{fc}{tr}{tc}")
    fname = "-".join(parts) + ".txt" if parts else "initial.txt"
    fpath = os.path.join(depth_dir, fname)
    with open(fpath, "w") as f:
        f.write(state.serialize())
    _save_to_buffer(state, best_cp)


def expand(proc, state, ply, max_ply, out_dirs, parent_score=0,
           score_window=SCORE_WINDOW, search_depth=SEARCH_DEPTH):
    """Recursively expand moves, saving positions at every depth level.

    Args:
        ply: current ply (0 = root, max_ply = leaf depth).
    """
    moves = get_scored_moves(proc, state, score_window=score_window,
                             search_depth=search_depth)
    # Weighted cap below root: weight ∝ 1/multipv (favours engine-recommended)
    if ply > 0 and len(moves) > MAX_BRANCH:
        # weight decay: 1/√mpv — top=1.0, 2nd=0.71, 10th=0.32 (~3:1)
        w = np.array([1.0 / (m[2] ** 0.5) for m in moves], dtype=np.float64)
        w /= w.sum()
        idx = np.random.choice(len(moves), size=MAX_BRANCH, replace=False, p=w)
        moves = [moves[i] for i in sorted(idx)]
    best_cp = max(s for _, s, _ in moves) if moves else parent_score

    # Only save balanced positions (abs(cp) <= window)
    if abs(best_cp) <= score_window:
        _save_position(state, best_cp, ply, out_dirs)

    if ply == max_ply:
        return 1

    if not moves:
        return 0

    count = 0
    for a, score, _ in moves:
        s = state.clone()
        s.apply_action(a)
        count += expand(proc, s, ply + 1, max_ply, out_dirs,
                        parent_score=score, score_window=score_window,
                        search_depth=search_depth)
    return count


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Generate opening book with Pikafish")
    ap.add_argument("--depth", type=int, default=DEPTH,
                    help=f"Ply depth to expand (default: {DEPTH})")
    ap.add_argument("--window", type=int, default=SCORE_WINDOW,
                    help=f"Score window in cp (default: {SCORE_WINDOW})")
    ap.add_argument("--search-depth", type=int, default=SEARCH_DEPTH,
                    help=f"Engine search depth per position (default: {SEARCH_DEPTH})")
    args = ap.parse_args()
    depth = args.depth
    score_window = args.window

    print(f"Config: depth={depth}  window={score_window}cp  search_depth={SEARCH_DEPTH}")
    print(f"Output: {OUT_DIR}")
    print(f"Starting pikafish: {PIKAFISH}")
    proc = subprocess.Popen(
        [PIKAFISH],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )
    try:
        _send(proc, "uci")
        _read_until(proc, "uciok")
        _send(proc, f"setoption name Hash value {HASH_MB}")
        _send(proc, f"setoption name Threads value {THREADS}")
        _send(proc, "isready")
        _read_until(proc, "readyok")

        # Wipe stale buffer
        if os.path.exists(BUFFER_DB):
            os.remove(BUFFER_DB)
        # Wipe old opening book positions
        import glob as _g, shutil as _sh
        for dpath in _g.glob(os.path.join(OUT_DIR, "depth_*")):
            _sh.rmtree(dpath)
        # Create output dirs per depth
        out_dirs = {}
        for d in range(depth + 1):
            dpath = os.path.join(OUT_DIR, f"depth_{d}")
            os.makedirs(dpath, exist_ok=True)
            out_dirs[d] = dpath

        initial = GAME.new_initial_state()
        total = expand(proc, initial, 0, depth, out_dirs,
                       score_window=score_window,
                       search_depth=args.search_depth)
        if _buffer is not None:
            _buffer.flush()
            _buffer.close()
            print(f"  Buffer: {BUFFER_DB}")
        print(f"\nDone.  Generated {total} leaf positions in {OUT_DIR}")
    finally:
        _send(proc, "quit")
        proc.wait(timeout=5)


if __name__ == "__main__":
    main()
