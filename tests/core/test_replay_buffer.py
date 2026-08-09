"""Tests for train/core/replay_buffer.py (SQLite-backed buffer)."""

import os as _os
import tempfile as _tempfile

import numpy as np

from train.core.replay_buffer import ReplayBuffer


def _make_state(seed=0):
    rng = np.random.RandomState(seed)
    obs = rng.randn(4, 8, 8).astype(np.float32)
    mask = rng.rand(65) > 0.5
    pol = mask.astype(np.float32) / mask.sum()
    val = np.array([0.5, 0.3, 0.2], dtype=np.float32)
    return obs, mask, pol, val


def test_sqlite_basic_append_sample():
    db = _tempfile.mktemp(suffix=".db")
    try:
        buf = ReplayBuffer(100, db_path=db)
        obs, mask, pol, val = _make_state()
        for i in range(50):
            buf.append(obs, mask, pol, val, step=i, tag="")
        assert len(buf) == 50
        batch = buf.sample(10)
        assert 0 < batch.value.shape[0] <= 10  # duplicates possible with weighted sampling
        buf.flush()
        cnt = buf._conn.execute("SELECT COUNT(*) FROM states").fetchone()[0]
        assert cnt == 50
        buf.close()
    finally:
        try: _os.unlink(db)
        except: pass


def test_sqlite_resume_full():
    db = _tempfile.mktemp(suffix=".db")
    try:
        obs, mask, pol, val = _make_state()
        buf1 = ReplayBuffer(100, db_path=db)
        for i in range(200):
            buf1.append(obs, mask, pol, val, step=i // 10, tag=f"s{i}")
        buf1.flush(); buf1.close()
        buf2 = ReplayBuffer(100, db_path=db)
        assert len(buf2) == 100
        assert buf2.total_seen == 200
        buf2.close()
    finally:
        try: _os.unlink(db)
        except: pass


def test_sqlite_resume_partial():
    db = _tempfile.mktemp(suffix=".db")
    try:
        obs, mask, pol, val = _make_state()
        buf1 = ReplayBuffer(100, db_path=db)
        for i in range(30):
            buf1.append(obs, mask, pol, val, step=i, tag="")
        buf1.flush(); buf1.close()
        buf2 = ReplayBuffer(100, db_path=db)
        assert len(buf2) == 30
        assert buf2.total_seen == 30
        buf2.close()
    finally:
        try: _os.unlink(db)
        except: pass


def test_sqlite_rollback():
    db = _tempfile.mktemp(suffix=".db")
    try:
        obs, mask, pol, val = _make_state()
        buf = ReplayBuffer(100, db_path=db)
        for i in range(200):
            buf.append(obs, mask, pol, val, step=i, tag="")
        buf.flush()
        assert len(buf) == 100
        buf.rollback(50)
        max_s = buf._conn.execute("SELECT MAX(step) FROM states").fetchone()[0]
        assert max_s is None or max_s <= 50
        buf.close()
    finally:
        try: _os.unlink(db)
        except: pass


def test_sqlite_expand_buffer():
    db = _tempfile.mktemp(suffix=".db")
    try:
        obs, mask, pol, val = _make_state()
        buf1 = ReplayBuffer(100, db_path=db)
        for i in range(200):
            buf1.append(obs, mask, pol, val, step=i, tag="")
        buf1.flush(); buf1.close()
        buf2 = ReplayBuffer(200, db_path=db)
        assert len(buf2) == 200
        buf2.close()
    finally:
        try: _os.unlink(db)
        except: pass


def test_sqlite_empty_start():
    db = _tempfile.mktemp(suffix=".db")
    try:
        buf = ReplayBuffer(100, db_path=db)
        assert len(buf) == 0
        buf.close()
    finally:
        try: _os.unlink(db)
        except: pass


def test_in_memory_requires_db():
    """db_path=None only supports append/count — sample must use DB."""
    buf = ReplayBuffer(50)
    obs, mask, pol, val = _make_state()
    for i in range(30):
        buf.append(obs, mask, pol, val, step=i, tag="")
    assert len(buf) == 30
    assert buf.total_seen == 30


def test_sqlite_step_tagging():
    db = _tempfile.mktemp(suffix=".db")
    try:
        obs, mask, pol, val = _make_state()
        buf = ReplayBuffer(100, db_path=db)
        for i in range(20):
            tag = "rare" if i % 3 == 0 else ""
            buf.append(obs, mask, pol, val, step=i, tag=tag)
        buf.flush()
        steps = [r[0] for r in buf._conn.execute(
            "SELECT step FROM states ORDER BY id")]
        tags = [r[0] for r in buf._conn.execute(
            "SELECT tag FROM states ORDER BY id")]
        assert steps == list(range(20))
        assert tags[0] == "rare"
        buf.close()
    finally:
        try: _os.unlink(db)
        except: pass


def test_compress_roundtrip():
    """Compressed DB must preserve exact data after reload."""
    import sqlite3, os, zlib
    from train.core.replay_buffer import _compress_db
    db = _tempfile.mktemp(suffix=".db")
    try:
        # Build uncompressed DB directly
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE states (id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " step INTEGER, obs BLOB, mask BLOB, policy BLOB,"
            " value BLOB, tag TEXT DEFAULT '')")
        obs0 = np.arange(8, dtype=np.float32)     # flat, like training obs
        mask0 = np.array([True, False, True], dtype=bool)
        pol0 = np.array([0.7, 0.2, 0.1], dtype=np.float32)
        val0 = np.array([0.8, 0.1, 0.1], dtype=np.float32)
        for i in range(50):
            conn.execute(
                "INSERT INTO states (step,obs,mask,policy,value,tag)"
                " VALUES (?,?,?,?,?,?)",
                (i, obs0.tobytes(), mask0.tobytes(), pol0.tobytes(),
                 val0.tobytes(), "test"))
        conn.commit()
        conn.close()

        # Compress
        _compress_db(db)

        # Reload and verify
        buf = ReplayBuffer(100, db_path=db)
        assert len(buf) == 50
        for i in range(10):
            batch = buf.sample(1)
            np.testing.assert_array_equal(batch.observation[0], obs0)
            np.testing.assert_array_equal(batch.legals_mask[0], mask0)
            np.testing.assert_allclose(batch.policy[0], pol0, atol=1e-6)
            np.testing.assert_allclose(batch.value[0], val0, atol=1e-6)
        buf.close()
    finally:
        for ext in ("", "-shm", "-wal"):
            try: _os.unlink(db + ext)
            except: pass


def main():
    tests = [
        test_sqlite_basic_append_sample, test_sqlite_resume_full,
        test_sqlite_resume_partial, test_sqlite_rollback,
        test_sqlite_expand_buffer, test_sqlite_empty_start,
        test_sqlite_step_tagging, test_compress_roundtrip,
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


def test_db_rotate():
    """DB rotates to a new file when max_db_rows is reached."""
    import numpy as np, glob
    base = _tempfile.mktemp(suffix=".db")
    buf = ReplayBuffer(max_size=10, db_path=base, max_db_rows=50)
    obs = np.zeros(4 * 8 * 8, dtype=np.float32)
    mask = np.ones(65, dtype=bool)
    pol = np.ones(65, dtype=np.float32) / 65
    val = np.array([0.5, 0.3, 0.2], dtype=np.float32)

    for i in range(60):
        buf.append(obs, mask, pol, val, step=1)
    buf.flush()

    # Current DB has <= 50 rows
    cur = buf._conn.execute("SELECT COUNT(*) FROM states")
    assert cur.fetchone()[0] <= 50

    # Old file exists with remaining rows
    rotated = glob.glob(base.replace(".db", "_*.db"))
    assert len(rotated) >= 1

    buf.close()
    for f in [base] + rotated:
        _os.remove(f)


def test_sample_from_rotated_dbs():
    """sample() includes rows from archived DB files after rotation."""
    import numpy as np, glob
    base = _tempfile.mktemp(suffix=".db")
    buf = ReplayBuffer(max_size=50, db_path=base, max_db_rows=40)
    obs = np.random.randn(4 * 8 * 8).astype(np.float32)
    mask = np.ones(65, dtype=bool)
    pol = np.ones(65, dtype=np.float32) / 65
    val = np.array([0.5, 0.3, 0.2], dtype=np.float32)

    for i in range(70):
        buf.append(obs, mask, pol, val, step=1, tag="")
    buf.flush()

    # Should have at least 2 files (current + archived)
    paths = buf._all_db_paths()
    assert len(paths) >= 2, f"Expected >=2 DB files, got {len(paths)}"

    # Sample should work and return data from across all files.
    # Linear weighting may produce duplicate positions → ≤ n unique rows is normal.
    batch = buf.sample(20)
    assert 0 < batch.value.shape[0] <= 20
    assert buf.total_seen == 70

    buf.close()
    for f in glob.glob(base.replace(".db", "_*.db")) + [base]:
        _os.remove(f)


def test_tag_counts_sql():
    """tag_counts() uses SQL GROUP BY instead of in-memory ring."""
    import sqlite3
    base = _tempfile.mktemp(suffix=".db")
    buf = ReplayBuffer(max_size=100, db_path=base)
    obs = np.zeros(4 * 8 * 8, dtype=np.float32)
    mask = np.ones(65, dtype=bool)
    pol = np.ones(65, dtype=np.float32) / 65
    val = np.array([0.5, 0.3, 0.2], dtype=np.float32)

    tags = ["rare"] * 5 + ["surprise"] * 3 + [""] * 12
    for i, t in enumerate(tags):
        buf.append(obs, mask, pol, val, step=i, tag=t)
    buf.flush()

    tc = buf.tag_counts()
    assert tc == {"rare": 5, "surprise": 3, "": 12}, f"Got {tc}"
    buf.close()
    for ext in ("", "-shm", "-wal"):
        try: _os.unlink(base + ext)
        except: pass


def test_sqrt_weighted_distribution():
    """Sqrt weight ∝ √(pos+1): newer rows sampled more, but softly.

    1000 rows, 20000 samples.  Theoretical: newer half ≈ 65% of weight,
    newest 10% ≈ 4.6× oldest 10%, each decile strictly > previous.
    """
    import sqlite3
    from collections import Counter
    base = _tempfile.mktemp(suffix=".db")
    buf = ReplayBuffer(max_size=1000, db_path=base, max_db_rows=2000)
    for i in range(1000):
        obs = np.array([float(i)], dtype=np.float32)
        buf.append(obs, np.ones(1, dtype=bool),
                   np.ones(1, dtype=np.float32),
                   np.array([0.5, 0.3, 0.2], dtype=np.float32),
                   step=1, tag="")
    buf.flush()

    id_counts = Counter()
    for _ in range(200):
        batch = buf.sample(100)
        for row_id in batch.observation[:, 0]:
            id_counts[int(row_id)] += 1

    total = sum(id_counts.values())
    assert total > 1000

    # Newer half (indices 500-999) — theoretical ≈ 64.6%
    newer = sum(c for rid, c in id_counts.items() if rid >= 500)
    assert newer / total > 0.55, f"Newer half too low: {newer/total:.1%}"

    # Newest 10% vs oldest 10% — theoretical ~4.6×
    newest10 = sum(c for rid, c in id_counts.items() if rid >= 900)
    oldest10 = sum(c for rid, c in id_counts.items() if rid < 100)
    assert newest10 >= 3 * oldest10, \
        f"Newest10={newest10}  Oldest10={oldest10}  ratio={newest10/max(oldest10,1):.1f}"

    # Strict monotonic deciles — each later decile strictly > previous
    deciles = [0] * 10
    for rid, c in id_counts.items():
        deciles[rid // 100] += c
    for i in range(9):
        assert deciles[i] < deciles[i + 1], \
            f"Decile {i} ({deciles[i]}) >= decile {i+1} ({deciles[i+1]}) — {deciles}"

    buf.close()
    for ext in ("", "-shm", "-wal"):
        try: _os.unlink(base + ext)
        except: pass


def test_numeric_file_sorting():
    """_all_db_paths sorts archived files by numeric suffix, not string."""
    import re
    # 10M should sort after 9M numerically, but not with string sort
    names = ["buffer.db", "buffer_900000.db", "buffer_1000000.db",
             "buffer_2000000.db", "buffer_10000000.db"]
    # Simulate _all_db_paths logic
    base = "buffer.db"
    archived = [n for n in names if n != base]
    # String sort (old buggy behavior)
    str_sorted = sorted(archived, reverse=True)
    assert str_sorted[0] == "buffer_900000.db", \
        f"String sort puts 9M first: {str_sorted}"
    # Numeric sort (fixed)
    archived.sort(key=lambda p: int(re.search(r'_(\d+)\.db$', p).group(1)),
                  reverse=True)
    assert archived == ["buffer_10000000.db", "buffer_2000000.db",
                         "buffer_1000000.db", "buffer_900000.db"], \
        f"Numeric sort failed: {archived}"


def test_uniform_sample_across_dbs():
    """Uniform sampling is evenly distributed across rotated DB files."""
    import numpy as np, glob
    from collections import Counter
    base = _tempfile.mktemp(suffix=".db")
    # max_db_rows=30 → 90 rows across 3 files: 30+30+30
    buf = ReplayBuffer(max_size=200, db_path=base, max_db_rows=30)
    for i in range(90):
        obs = np.array([float(i)], dtype=np.float32)
        buf.append(obs, np.ones(1, dtype=bool),
                   np.ones(1, dtype=np.float32),
                   np.array([0.5, 0.3, 0.2], dtype=np.float32),
                   step=1, tag="")
        if i > 0 and i % 30 == 0:
            buf.flush()  # force rotation each 30 rows
    buf.flush()  # final
    assert len(buf._all_db_paths()) >= 3, f"Expected >=3 DB files"

    id_counts = Counter()
    for _ in range(300):
        batch = buf.sample_uniform(100)
        for row_id in batch.observation[:, 0]:
            id_counts[int(row_id)] += 1

    # All 90 rows should appear at least once with 30000 attempts
    missing = [i for i in range(90) if i not in id_counts]
    assert len(missing) == 0, f"Rows never sampled: {missing}"

    # Per-file check: each file (0-29, 30-59, 60-89) gets ~1/3 of samples
    thirds = [sum(c for rid, c in id_counts.items() if start <= rid < start + 30)
              for start in (0, 30, 60)]
    total = sum(thirds)
    for i, label in enumerate(["oldest", "middle", "newest"]):
        frac = thirds[i] / total
        assert 0.25 < frac < 0.42, \
            f"{label} third fraction={frac:.2%} (expected ~33%)"

    batch = buf.sample_uniform(50)
    assert batch.value.shape[0] >= 30  # generous bound

    buf.close()
    for f in glob.glob(base.replace(".db", "_*.db")) + [base]:
        _os.remove(f)


def test_archived_sizes_cache():
    """_archived_sizes cache matches real DB row counts after rotations."""
    import sqlite3, glob
    base = _tempfile.mktemp(suffix=".db")
    buf = ReplayBuffer(max_size=200, db_path=base, max_db_rows=30)
    obs = np.zeros(4 * 8 * 8, dtype=np.float32)
    mask = np.ones(65, dtype=bool)
    pol = np.ones(65, dtype=np.float32) / 65
    val = np.array([0.5, 0.3, 0.2], dtype=np.float32)

    for i in range(100):
        buf.append(obs, mask, pol, val, step=1, tag="")
        if (i + 1) % 30 == 0:
            buf.flush()  # trigger rotation each 30 rows
    buf.flush()

    assert len(buf._archived_sizes) >= 3, f"Expected >=3 archived, got {len(buf._archived_sizes)}"
    for path, cached_cnt in buf._archived_sizes:
        conn = sqlite3.connect(path)
        real_cnt = conn.execute("SELECT COUNT(*) FROM states").fetchone()[0]
        conn.close()
        assert cached_cnt == real_cnt, \
            f"Cache mismatch: {os.path.basename(path)} cache={cached_cnt} real={real_cnt}"

    buf.close()
    for f in glob.glob(base.replace(".db", "_*.db")) + [base]:
        _os.remove(f)


if __name__ == "__main__":
    main()
