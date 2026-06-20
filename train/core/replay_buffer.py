"""SQLite-backed FIFO ring buffer for AlphaZero samples.

Each state is persisted immediately to SQLite with its training step,
solving crash-resilience and checkpoint-alignment in one shot.
An in-memory ring buffer mirrors the most recent *max_size* states
for fast GPU sampling.
"""

import os
import sqlite3
import zlib

import numpy as np

from train.core.types import TrainInput

_WAL_PRAGMAS = ("PRAGMA journal_mode=WAL;", "PRAGMA synchronous=NORMAL;")


from train.core.position_hash import hash_obs


def _pack(arr):
    return zlib.compress(np.asarray(arr, dtype=np.float32).tobytes())


def _pack_bool(arr):
    return zlib.compress(np.asarray(arr, dtype=bool).tobytes())


def _unpack(raw, dtype):
    try:
        return np.frombuffer(zlib.decompress(raw), dtype=dtype)
    except zlib.error:
        return np.frombuffer(raw, dtype=dtype)  # old uncompressed format


class ReplayBuffer:
    """SQLite-persisted FIFO ring buffer.

    Args:
        max_size: in-memory ring capacity (also limits how many recent
                  states are loaded from DB on resume).
        db_path: path to SQLite file.  If None, operates in-memory only
                 (backward compat for tests).
    """

    def __init__(self, max_size: int, db_path: str = None,
                 max_db_rows: int = 1_000_000, recent_db_rows: int = 0,
                 game_name: str = ""):
        self._max_size = max_size
        self._db_path = db_path
        self._max_db_rows = max_db_rows
        self._recent_db_rows = recent_db_rows
        self._obs = None
        self._masks = None
        self._policies = None
        self._values = None
        self._tags = None           # ring-buffer tags
        self._index = 0             # total states ever appended
        self._size = 0              # ring occupancy
        self._pending = 0           # unflushed DB inserts
        self._game_name = game_name
        self._recent_pending = 0

        if db_path:
            self._conn = sqlite3.connect(db_path, timeout=30)
            for p in _WAL_PRAGMAS:
                self._conn.execute(p)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS states ("
                "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "  step INTEGER NOT NULL,"
                "  obs BLOB NOT NULL,"
                "  mask BLOB NOT NULL,"
                "  policy BLOB NOT NULL,"
                "  value BLOB NOT NULL,"
                "  tag TEXT NOT NULL DEFAULT ''"
                ")"
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_step ON states(step)")
            self._conn.commit()
            self._load_ring_from_db()
            # Separate small DB for the most recent N rows (fast local download)
            if recent_db_rows > 0:
                rpath = db_path.replace(".db", "_recent.db")
                self._recent_conn = sqlite3.connect(rpath, timeout=30)
                for p in _WAL_PRAGMAS:
                    self._recent_conn.execute(p)
                self._recent_conn.execute(
                    "CREATE TABLE IF NOT EXISTS states ("
                    "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
                    "  step INTEGER NOT NULL,"
                    "  obs BLOB NOT NULL,"
                    "  mask BLOB NOT NULL,"
                    "  policy BLOB NOT NULL,"
                    "  value BLOB NOT NULL,"
                    "  tag TEXT NOT NULL DEFAULT ''"
                    ")"
                )
                self._recent_conn.commit()
            else:
                self._recent_conn = None
        else:
            self._conn = None
            self._recent_conn = None

    # ── Public API ──────────────────────────────────────────────────────────

    def append(self, obs: np.ndarray, mask: np.ndarray,
               policy: np.ndarray, value, tag: str = "", step: int = 0):
        """Insert one state.  *step* is the training step that produced it."""
        if self._obs is None:
            self._obs = np.empty((self._max_size, *obs.shape), dtype=np.float32)
            self._masks = np.empty((self._max_size, *mask.shape), dtype=bool)
            self._policies = np.empty((self._max_size, *policy.shape),
                                       dtype=np.float32)
            self._values = np.empty((self._max_size, 3), dtype=np.float32)
            self._tags = np.empty((self._max_size,), dtype=object)

        # In-memory ring
        idx = self._index % self._max_size
        self._obs[idx] = obs.astype(np.float32)
        self._masks[idx] = mask
        self._policies[idx] = policy.astype(np.float32)
        self._values[idx] = np.asarray(value, dtype=np.float32)
        self._tags[idx] = tag
        self._index += 1
        self._size = min(self._size + 1, self._max_size)

        # SQLite — batch flush every 256 appends
        if self._conn:
            self._conn.execute(
                "INSERT INTO states (step, obs, mask, policy, value, tag) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (step, _pack(obs), _pack_bool(mask), _pack(policy),
                 _pack(value), tag))
            self._pending += 1
            if self._pending >= 256:
                self._conn.commit()
                self._pending = 0
                self._rotate_db()
        # Recent-only DB (small, fast to download)
        if self._recent_conn:
            self._recent_conn.execute(
                "INSERT INTO states (step, obs, mask, policy, value, tag) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (step, _pack(obs), _pack_bool(mask), _pack(policy),
                 _pack(value), tag))
            self._recent_pending += 1
            if self._recent_pending >= 256:
                self._recent_conn.execute(
                    "DELETE FROM states WHERE id NOT IN ("
                    "  SELECT id FROM states ORDER BY id DESC LIMIT ?)",
                    (self._recent_db_rows,))
                self._recent_conn.commit()
                self._recent_pending = 0

    def sample(self, n: int) -> TrainInput:
        """Random sample from the in-memory ring buffer."""
        indices = np.random.randint(0, self._size, size=n)
        return TrainInput(
            observation=self._obs[indices],
            legals_mask=self._masks[indices],
            policy=self._policies[indices],
            value=self._values[indices],
        )

    def tag_counts(self) -> dict:
        if self._tags is None:
            return {}
        tags = self._tags[:self._size]
        valid = [str(t) for t in tags if t is not None]
        if not valid:
            return {}
        unique, counts = np.unique(valid, return_counts=True)
        return {str(k): int(v) for k, v in zip(unique, counts)}

    def flush(self):
        """Commit any pending SQLite writes (call before shutdown)."""
        if self._conn and self._pending > 0:
            self._conn.commit()
            self._pending = 0
        if self._conn:
            self._rotate_db()
        if self._recent_conn and self._recent_pending > 0:
            self._recent_conn.execute(
                "DELETE FROM states WHERE id NOT IN ("
                "  SELECT id FROM states ORDER BY id DESC LIMIT ?)",
                (self._recent_db_rows,))
            self._recent_conn.commit()
            self._recent_pending = 0

    def _rotate_db(self, db_path=None):
        """If current DB exceeds max_db_rows, archive it and start a new one."""
        if db_path is None:
            db_path = self._db_path
        cur = self._conn.execute("SELECT COUNT(*) FROM states")
        if cur.fetchone()[0] < self._max_db_rows:
            return
        self._conn.close()
        suffix = self._index - self._max_db_rows  # approx row count in archived file
        rotated = db_path.replace(".db", f"_{suffix}.db")
        os.rename(db_path, rotated)
        self._conn = sqlite3.connect(db_path, timeout=30)
        for p in _WAL_PRAGMAS:
            self._conn.execute(p)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS states ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  step INTEGER NOT NULL,"
            "  obs BLOB NOT NULL,"
            "  mask BLOB NOT NULL,"
            "  policy BLOB NOT NULL,"
            "  value BLOB NOT NULL,"
            "  tag TEXT NOT NULL DEFAULT ''"
            ")"
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_step ON states(step)")
        self._conn.commit()

    def _all_db_paths(self):
        """Return all buffer DB paths (current + archived), newest first."""
        import glob as _glob
        base = self._db_path
        archived = sorted(_glob.glob(base.replace(".db", "_*.db")), reverse=True)
        if os.path.exists(base):
            return [base] + archived
        return archived

    def _load_ring_from_db(self):
        """Fill the in-memory ring from the most recent DB rows across all files."""
        paths = self._all_db_paths()
        rows = []
        for p in paths:
            conn = sqlite3.connect(p)
            cur = conn.execute(
                "SELECT obs, mask, policy, value, tag FROM states "
                "ORDER BY id DESC LIMIT ?", (self._max_size,))
            rows.extend(cur.fetchall())
            conn.close()
            if len(rows) >= self._max_size:
                break
        rows = rows[:self._max_size]
        rows.reverse()  # chronological order

        n = len(rows)
        if n == 0:
            self._index = 0
            self._size = 0
            return
        first_obs = _unpack(rows[0][0], np.float32)
        first_mask = _unpack(rows[0][1], bool)
        first_pol = _unpack(rows[0][2], np.float32)

        self._obs = np.empty((self._max_size, *first_obs.shape), dtype=np.float32)
        self._masks = np.empty((self._max_size, *first_mask.shape), dtype=bool)
        self._policies = np.empty((self._max_size, *first_pol.shape), dtype=np.float32)
        self._values = np.empty((self._max_size, 3), dtype=np.float32)
        self._tags = np.empty((self._max_size,), dtype=object)

        for i, (obs_b, mask_b, pol_b, val_b, tag) in enumerate(rows):
            self._obs[i] = _unpack(obs_b, np.float32)
            self._masks[i] = _unpack(mask_b, bool)
            self._policies[i] = _unpack(pol_b, np.float32)
            self._values[i] = _unpack(val_b, np.float32)
            self._tags[i] = tag

        self._size = n
        # Recover _index: total rows across all files
        total = 0
        for p in paths:
            conn = sqlite3.connect(p)
            cur = conn.execute("SELECT COUNT(*) FROM states")
            total += cur.fetchone()[0]
            conn.close()
        self._index = max(total, n)

    def rollback(self, target_step: int):
        """Delete states with step > target_step from all DB files, reload ring."""
        if not self._conn:
            return
        self.flush()
        # Only touch files that might have step > target_step.
        # Archived files are named buffer_<first_index>.db — skip if all rows
        # in the file predate the target step.
        for p in self._all_db_paths():
            conn = sqlite3.connect(p)
            cur = conn.execute("SELECT MAX(step) FROM states")
            max_s = cur.fetchone()[0]
            if max_s is not None and max_s <= target_step:
                conn.close()
                continue
            conn.execute("DELETE FROM states WHERE step > ?", (target_step,))
            conn.commit()
            conn.close()
        if self._recent_conn:
            self._recent_conn.execute("DELETE FROM states")
            self._recent_conn.commit()
        self._obs = None
        self._load_ring_from_db()

    def close(self):
        """Close the database connection."""
        self.flush()
        if self._conn:
            self._conn.close()
            self._conn = None
        if self._recent_conn:
            self._recent_conn.close()
            self._recent_conn = None

    # ── Properties ──────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self._size

    @property
    def total_seen(self) -> int:
        return self._index

    @property
    def unique_states(self) -> int:
        if self._obs is None or self._size == 0:
            return 0
        seen = set()
        for i in range(self._size):
            seen.add(hash_obs(self._obs[i], self._game_name))
        return len(seen)


# ── Buffer converter (old scalar → new WDL) ──────────────────────────────────

def _convert_buffer(src_path: str, dst_path: str = None):
    """Convert old scalar buffer to WDL format.  Writes in-place if no dst."""
    data = np.load(src_path)
    vals = data["values"]
    if vals.ndim > 1:
        data.close()
        print(f"[convert] Already WDL format ({vals.shape[1]}-dim), skipping.")
        return

    wdl = np.zeros((vals.shape[0], 3), dtype=np.float32)
    v = vals.astype(np.float32)
    wdl[:, 0] = np.maximum(v, 0)
    wdl[:, 1] = 1.0 - np.abs(v)
    wdl[:, 2] = np.maximum(-v, 0)

    obs = data["obs"].copy()
    masks = data["masks"].copy()
    policies = data["policies"].copy()
    tags = data.get("tags", np.array([""] * vals.shape[0]))
    idx = int(data.get("index", vals.shape[0]))
    sz = int(data.get("size", vals.shape[0]))
    data.close()

    out = dst_path or src_path
    tmp = out + ".converting.npz"
    np.savez_compressed(tmp, obs=obs, masks=masks, policies=policies,
                        values=wdl, tags=tags, index=idx, size=sz)
    os.replace(tmp, out)
    print(f"[convert] {vals.shape[0]} states: scalar → WDL → {out}")


def _compress_db(db_path: str):
    """In-place convert uncompressed DB rows to zlib-compressed format."""
    print(f"[compress] Opening {db_path} ...")
    conn = sqlite3.connect(db_path)
    for p in _WAL_PRAGMAS:
        conn.execute(p)
    cur = conn.execute("SELECT COUNT(*) FROM states")
    total = cur.fetchone()[0]
    print(f"[compress] {total} rows — scanning for uncompressed ...")

    # Test first row to check if already compressed
    test = conn.execute("SELECT obs FROM states LIMIT 1").fetchone()
    try:
        zlib.decompress(test[0])
        print("[compress] Already compressed — nothing to do.")
        conn.close()
        return
    except zlib.error:
        pass

    updated = 0
    for row in conn.execute("SELECT id, obs, mask, policy, value FROM states"):
        rid, obs_b, mask_b, pol_b, val_b = row
        new_obs = sqlite3.Binary(zlib.compress(obs_b))
        new_mask = sqlite3.Binary(zlib.compress(mask_b))
        new_pol = sqlite3.Binary(zlib.compress(pol_b))
        new_val = sqlite3.Binary(zlib.compress(val_b))
        conn.execute(
            "UPDATE states SET obs=?, mask=?, policy=?, value=? WHERE id=?",
            (new_obs, new_mask, new_pol, new_val, rid))
        updated += 1
        if updated % 10000 == 0:
            conn.commit()
            print(f"[compress] {updated}/{total} ...")
    conn.commit()
    conn.execute("VACUUM")
    conn.close()
    print(f"[compress] Done — {updated} rows compressed + VACUUM.")


if __name__ == "__main__":
    _compress_db(r"C:\Users\shouk\othello_train\cloud_wdl_db\buffer.db")
