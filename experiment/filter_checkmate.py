"""Extract checkmate (proven win/loss) rows from buffer DBs into new DBs.

Usage:
    python -m experiment.filter_checkmate --src SRC_DIR --dst DST_DIR
    python -m experiment.filter_checkmate --src cloud_pre --dst cloud_checkmate

Output: buffer_checkmate_NNNNNNN.db files under --dst, one per source DB.
"""

import argparse
import os
import re
import sqlite3
import sys
import zlib
from glob import glob

import numpy as np

_sys_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _sys_root not in sys.path:
    sys.path.insert(0, _sys_root)

from train.core.replay_buffer import _unpack

WAL_PRAGMAS = ("PRAGMA journal_mode=WAL;", "PRAGMA synchronous=NORMAL;")


def is_checkmate(value_blob: bytes) -> bool:
    """Return True if value is a proven win or loss (w ≈ 1 or l ≈ 1)."""
    val = _unpack(value_blob, np.float32)
    return val[0] >= 0.9999 or val[2] >= 0.9999


def filter_db(src_path: str, dst_path: str, max_rows: int = 2_000_000):
    """Extract checkmate rows from *src_path* into *dst_path*, rotating at max_rows."""
    src = sqlite3.connect(src_path)
    total = src.execute("SELECT COUNT(*) FROM states").fetchone()[0]
    cur = src.execute("SELECT id, step, obs, mask, policy, value, tag FROM states ORDER BY id")

    seen_hashes = set()
    dup_count = 0
    dst_conn = None
    dst_suffix = 0
    dst_count = 0
    match_count = 0
    file_count = 0

    def _open_dst():
        nonlocal dst_conn, dst_count, dst_suffix
        if dst_conn:
            dst_conn.commit()
            dst_conn.close()
        fname = f"buffer_checkmate_{dst_suffix}.db"
        path = os.path.join(dst_path, fname)
        dst_conn = sqlite3.connect(path)
        for p in WAL_PRAGMAS:
            dst_conn.execute(p)
        dst_conn.execute(
            "CREATE TABLE IF NOT EXISTS states ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  step INTEGER NOT NULL,"
            "  obs BLOB NOT NULL, mask BLOB NOT NULL,"
            "  policy BLOB NOT NULL, value BLOB NOT NULL,"
            "  tag TEXT NOT NULL DEFAULT ''"
            ")"
        )
        dst_conn.execute("CREATE INDEX IF NOT EXISTS idx_step ON states(step)")
        dst_conn.commit()
        dst_count = 0
        dst_suffix += 1

    os.makedirs(dst_path, exist_ok=True)
    _open_dst()

    for rid, step, obs, mask, policy, value, tag in cur:
        if is_checkmate(value):
            h = hash(obs)
            if h in seen_hashes:
                dup_count += 1
                continue
            seen_hashes.add(h)
            dst_conn.execute(
                "INSERT INTO states (step, obs, mask, policy, value, tag) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (step, obs, mask, policy, value, tag))
            match_count += 1
            dst_count += 1
            if dst_count % 10000 == 0:
                dst_conn.commit()
            if dst_count >= max_rows:
                _open_dst()

        if (rid + 1) % 500000 == 0:
            print(f"  {rid + 1:>8d}/{total}  checkmate={match_count}", flush=True)

    src.close()
    if dst_conn:
        dst_conn.commit()
        dst_conn.close()

    print(f"\n  Done: {match_count} checkmate rows ({dup_count} dups skipped)"
          f" from {os.path.basename(src_path)}"
          f" → {file_count + 1} output file(s)")


def main():
    parser = argparse.ArgumentParser(
        description="Extract proven win/loss rows from buffer DBs")
    parser.add_argument("--src", type=str, required=True,
                        help="Source directory containing buffer_*.db files")
    parser.add_argument("--dst", type=str, required=True,
                        help="Destination directory for filtered DBs")
    parser.add_argument("--max-rows", type=int, default=2_000_000,
                        help="Max rows per output DB file before rotation")
    args = parser.parse_args()

    src_files = sorted(
        [f for f in glob(os.path.join(args.src, "buffer_*.db"))
         if re.search(r'buffer_(\d+)\.db', f)],
        key=lambda f: int(re.search(r'buffer_(\d+)\.db', f).group(1)))
    if not src_files:
        print(f"No buffer_*.db files found in {args.src}")
        return

    print(f"Found {len(src_files)} buffer DB files in {args.src}")
    total_all = 0
    for p in src_files:
        fname = os.path.basename(p)
        conn = sqlite3.connect(p)
        total = conn.execute("SELECT COUNT(*) FROM states").fetchone()[0]
        conn.close()
        total_all += total
        print(f"  {fname}: {total:>10,} rows")
    print(f"  Total: {total_all:>10,} rows\n")

    for p in src_files:
        print(f"Processing {os.path.basename(p)} ...")
        filter_db(p, args.dst, args.max_rows)


if __name__ == "__main__":
    main()
