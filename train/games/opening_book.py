"""Opening book base class — loads serialized states from a directory.

Each file is a text file containing state.serialize() output, compatible
with the format saved by xiangqi_ui.
"""

import os
import glob


class OpeningBook:
    """Collection of pre-recorded opening positions."""

    def __init__(self, game, dir_path: str):
        self._game = game
        self._states = []
        if not dir_path or not os.path.isdir(dir_path):
            return
        files = sorted(glob.glob(os.path.join(dir_path, "*.txt")))
        for fpath in files:
            try:
                with open(fpath) as f:
                    raw = f.read().strip()
                s = game.deserialize_state(raw)
                if not s.is_terminal():
                    self._states.append(s)
            except Exception:
                pass

    def __len__(self):
        return len(self._states)

    def __bool__(self):
        return len(self._states) > 0

    def sample(self, rng):
        """Return a random cloned opening state, or None if empty."""
        if not self._states:
            return None
        idx = rng.randint(0, len(self._states))
        return self._states[idx].clone()
