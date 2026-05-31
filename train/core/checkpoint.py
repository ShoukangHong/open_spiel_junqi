"""Checkpoint utilities — game-agnostic."""

import os


def find_latest_checkpoint(path):
    """Scan *path* for checkpoint-<N>.pt files, return the highest N."""
    best = 0
    for name in os.listdir(path):
        if name.startswith("checkpoint-") and name.endswith(".pt"):
            try:
                s = int(name.split("-")[1].split(".")[0])
                if s > best:
                    best = s
            except (ValueError, IndexError):
                pass
    return best
