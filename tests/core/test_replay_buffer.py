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
        assert batch.value.shape == (10, 3)
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


def test_sqlite_in_memory_mode():
    buf = ReplayBuffer(50)
    obs, mask, pol, val = _make_state()
    for i in range(30):
        buf.append(obs, mask, pol, val, step=i, tag="")
    assert len(buf) == 30
    assert buf._conn is None
    batch = buf.sample(5)
    assert batch.value.shape == (5, 3)


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
        test_sqlite_in_memory_mode, test_sqlite_step_tagging,
        test_compress_roundtrip,
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


if __name__ == "__main__":
    main()
