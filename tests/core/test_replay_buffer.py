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


def main():
    tests = [
        test_sqlite_basic_append_sample, test_sqlite_resume_full,
        test_sqlite_resume_partial, test_sqlite_rollback,
        test_sqlite_expand_buffer, test_sqlite_empty_start,
        test_sqlite_in_memory_mode, test_sqlite_step_tagging,
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
