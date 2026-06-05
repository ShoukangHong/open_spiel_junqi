"""Tests for train/core/checkpoint.py."""

import os

from train.core.checkpoint import find_latest_checkpoint


def test_find_latest_empty():
    dirpath = os.path.join(".", "empty_checkpoint_test")
    os.makedirs(dirpath, exist_ok=True)
    try:
        assert find_latest_checkpoint(dirpath) == 0
    finally:
        os.rmdir(dirpath)


def test_find_latest_single():
    dirpath = os.path.join(".", "single_checkpoint_test")
    os.makedirs(dirpath, exist_ok=True)
    try:
        open(os.path.join(dirpath, "checkpoint-5.pt"), "w").close()
        assert find_latest_checkpoint(dirpath) == 5
    finally:
        for f in os.listdir(dirpath):
            os.remove(os.path.join(dirpath, f))
        os.rmdir(dirpath)


def test_find_latest_multiple():
    dirpath = os.path.join(".", "multi_checkpoint_test")
    os.makedirs(dirpath, exist_ok=True)
    try:
        for step in [10, 30, 20]:
            open(os.path.join(dirpath, f"checkpoint-{step}.pt"), "w").close()
        assert find_latest_checkpoint(dirpath) == 30
    finally:
        for f in os.listdir(dirpath):
            os.remove(os.path.join(dirpath, f))
        os.rmdir(dirpath)


def test_find_latest_ignores_other_files():
    dirpath = os.path.join(".", "mixed_checkpoint_test")
    os.makedirs(dirpath, exist_ok=True)
    try:
        open(os.path.join(dirpath, "checkpoint-10.pt"), "w").close()
        open(os.path.join(dirpath, "checkpoint-abc.pt"), "w").close()
        open(os.path.join(dirpath, "other.txt"), "w").close()
        assert find_latest_checkpoint(dirpath) == 10
    finally:
        for f in os.listdir(dirpath):
            os.remove(os.path.join(dirpath, f))
        os.rmdir(dirpath)


def main():
    tests = [
        test_find_latest_empty, test_find_latest_single,
        test_find_latest_multiple, test_find_latest_ignores_other_files,
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
