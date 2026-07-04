"""SQLite-backed replay buffer for AlphaZero samples — disk only, no memory ring.

Append writes directly to SQLite.  Sample picks random rows via ID lookup.
Buffer size is limited only by disk — no in-memory numpy arrays.
"""

import os
import sqlite3
import zlib

import numpy as np

from train.core.types import TrainInput

_WAL_PRAGMAS = ("PRAGMA journal_mode=WAL;", "PRAGMA synchronous=NORMAL;")


def _pack(arr):
    return zlib.compress(np.asarray(arr, dtype=np.float32).tobytes())


def _pack_bool(arr):
    return zlib.compress(np.asarray(arr, dtype=bool).tobytes())


def _unpack(raw, dtype):
    try:
        return np.frombuffer(zlib.decompress(raw), dtype=dtype)
    except zlib.error:
        return np.frombuffer(raw, dtype=dtype)


class ReplayBuffer:
    """Disk-only SQLite replay buffer.

    Args:
        max_size: deprecated (kept for API compat) — buffer is unbounded.
        db_path: SQLite file path.
        max_db_rows: rotate DB when this many rows accumulate.
        recent_db_rows: if >0, maintain a small _recent.db with latest N rows.
    """

    def __init__(self, max_size: int = 500_000, db_path: str = None,
                 max_db_rows: int = 1_000_000, recent_db_rows: int = 0):
        self._max_size = max_size
        self._db_path = db_path
        self._max_db_rows = max_db_rows
        self._recent_db_rows = recent_db_rows
        self._total = 0              # total rows ever appended (across rotations)
        self._pending = 0            # unflushed DB inserts
        self._recent_pending = 0
        self._sample_rng = np.random.RandomState()
        self._stats_total = 0
        self._stats_hashes = set()
        self._diag_files = {}        # cumulative file→count across all samples
        self._diag_step_min = 10**9  # min training step seen this step
        self._diag_step_max = 0      # max training step seen this step

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
            self._sync_total()
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

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _sync_total(self):
        """Recalculate _total from all DB files (for resume / rollback)."""
        if not self._db_path:
            return
        self._total = 0
        for p in self._all_db_paths():
            try:
                conn = sqlite3.connect(p)
                cur = conn.execute("SELECT COUNT(*) FROM states")
                self._total += cur.fetchone()[0]
                conn.close()
            except Exception:
                pass

    def _all_db_paths(self):
        import glob as _glob, re as _re
        base = self._db_path
        archived = [p for p in _glob.glob(base.replace(".db", "_*.db"))
                    if _re.search(r'_(\d+)\.db$', p)]
        # Sort by numeric suffix — string sort fails at 10M (9 > 1)
        archived.sort(key=lambda p: int(_re.search(r'_(\d+)\.db$', p).group(1)),
                      reverse=True)
        if os.path.exists(base):
            return [base] + archived
        return archived

    def _count_current(self):
        cur = self._conn.execute("SELECT COUNT(*) FROM states")
        return cur.fetchone()[0]

    # ── Public API ──────────────────────────────────────────────────────────

    def append(self, obs: np.ndarray, mask: np.ndarray,
               policy: np.ndarray, value, tag: str = "", step: int = 0):
        """Insert one state.  *step* is the training step that produced it."""
        self._total += 1
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
        """Default: linear-weighted sampling."""
        return self.sample_weighted(n)

    def sample_weighted(self, n: int) -> TrainInput:
        """Linear-weighted random sample — newer rows have higher probability.

        Weight ∝ position within the window (0 = oldest, w-1 = newest).
        Samples via inverse-CDF of pos² distribution.
        """
        if self._conn is None:
            raise RuntimeError("sample() requires a DB-backed buffer")
        w = min(self._total, self._max_size)
        if w == 0:
            raise RuntimeError("buffer is empty")
        u = self._sample_rng.random(n).astype(np.float64)
        pos = np.clip((np.sqrt(u * w * (w + 1) + 0.25) - 0.5).astype(np.int64),
                       0, w - 1)
        start_abs = max(0, self._total - w)
        abs_ids = (start_abs + pos + 1).tolist()
        return self._sample_by_ids(abs_ids)

    def sample_uniform(self, n: int) -> TrainInput:
        """Uniform random sample from the buffer window."""
        if self._conn is None:
            raise RuntimeError("sample() requires a DB-backed buffer")
        w = min(self._total, self._max_size)
        if w == 0:
            raise RuntimeError("buffer is empty")
        start_abs = max(0, self._total - w)
        abs_ids = (start_abs
                   + self._sample_rng.randint(1, w + 1, size=n)).tolist()
        return self._sample_by_ids(abs_ids)

    def _sample_by_ids(self, abs_ids):
        """Query rows by global IDs across all DB files, unpack, track stats."""
        paths = list(reversed(self._all_db_paths()))
        file_sizes = []
        for p in paths:
            conn = self._conn if p == self._db_path else None
            close_after = False
            if conn is None:
                conn = sqlite3.connect(p)
                close_after = True
            try:
                cnt = conn.execute("SELECT COUNT(*) FROM states").fetchone()[0]
                file_sizes.append((p, cnt))
            finally:
                if close_after:
                    conn.close()

        file_ofs = {}
        for a_id in abs_ids:
            remain = a_id
            for p, cnt in file_sizes:
                if remain <= cnt:
                    file_ofs.setdefault(p, []).append(remain)
                    break
                remain -= cnt

        rows = []
        file_counts = []
        step_vals = []
        temp_conns = []
        for p, local_ids in file_ofs.items():
            conn = self._conn if p == self._db_path else sqlite3.connect(p)
            if p != self._db_path:
                temp_conns.append(conn)
            placeholders = ",".join("?" for _ in local_ids)
            file_rows = conn.execute(
                f"SELECT obs, mask, policy, value, step FROM states "
                f"WHERE id IN ({placeholders})",
                local_ids).fetchall()
            rows.extend(file_rows)
            fname = os.path.basename(p)
            file_counts.append((fname, len(file_rows)))
            step_vals.extend(r[4] for r in file_rows)

        for c in temp_conns:
            try: c.close()
            except: pass

        obs_l, mask_l, pol_l, val_l = [], [], [], []
        for obs_b, mask_b, pol_b, val_b, step_b in rows:
            obs_l.append(_unpack(obs_b, np.float32))
            mask_l.append(_unpack(mask_b, bool))
            pol_l.append(_unpack(pol_b, np.float32))
            val_l.append(_unpack(val_b, np.float32))
            self._stats_total += 1
            self._stats_hashes.add(hash(obs_b))
        # Accumulate diagnostic stats across all sample() calls in this step
        for fname, cnt in file_counts:
            self._diag_files[fname] = self._diag_files.get(fname, 0) + cnt
        if step_vals:
            self._diag_step_min = min(self._diag_step_min, min(step_vals))
            self._diag_step_max = max(self._diag_step_max, max(step_vals))

        return TrainInput(
            observation=np.stack(obs_l),
            legals_mask=np.stack(mask_l),
            policy=np.stack(pol_l),
            value=np.stack(val_l),
        )

    def sample_diag(self) -> str:
        """One-line diagnostic — cumulative file/step stats, resets on read."""
        if not self._diag_files:
            return "diag=(no data)"
        files = " ".join(f"{f}={c}" for f, c in sorted(self._diag_files.items()))
        smin, smax = self._diag_step_min, self._diag_step_max
        step_r = f"step[{smin}-{smax}]" if smin <= smax else "step[empty]"
        self._diag_files.clear()
        self._diag_step_min = 10**9
        self._diag_step_max = 0
        return f"diag=({step_r}  files: {files})"

    _STATS_WINDOW = 50000

    def tag_counts(self) -> dict:
        """Tag distribution over the latest {_STATS_WINDOW} rows (current DB)."""
        if not self._conn:
            return {}
        cur = self._conn.execute(
            "SELECT tag, COUNT(*) FROM ("
            "  SELECT tag FROM states ORDER BY id DESC LIMIT ?"
            ") GROUP BY tag", (self._STATS_WINDOW,))
        return {str(k): int(v) for k, v in cur.fetchall()}

    def flush(self):
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

    def _rotate_db(self):
        if self._db_path is None:
            return
        if self._count_current() < self._max_db_rows:
            return
        self._conn.close()
        suffix = self._total  # monotonic, never conflicts with existing archives
        rotated = self._db_path.replace(".db", f"_{suffix}.db")
        os.rename(self._db_path, rotated)
        self._conn = sqlite3.connect(self._db_path, timeout=30)
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

    def rollback(self, target_step: int):
        if not self._conn:
            return
        self.flush()
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
        self._sync_total()

    def close(self):
        self.flush()
        if self._conn:
            self._conn.close()
            self._conn = None
        if self._recent_conn:
            self._recent_conn.close()
            self._recent_conn = None

    # ── Properties ──────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return min(self._total, self._max_size)

    @property
    def total_seen(self) -> int:
        return self._total

    @property
    def recent_unique_ratio(self) -> float:
        """Fraction of unique samples drawn this step (resets on read)."""
        if self._stats_total == 0:
            return 1.0
        r = len(self._stats_hashes) / self._stats_total
        self._stats_total = 0
        self._stats_hashes.clear()
        return r

    @property
    def unique_states(self) -> int:
        """Unique samples this step (estimated from sampled ratio)."""
        if self._stats_total == 0:
            return 0
        return len(self._stats_hashes)


# ── Buffer maintenance ───────────────────────────────────────────────────────

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
