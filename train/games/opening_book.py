"""Opening book — stores serialized action sequences, reconstructs on demand.

Each file is a text file containing state.serialize() output.  Only the raw
strings are kept in memory; pyspiel States are reconstructed at sample time
via game.deserialize_state().
"""

import os
import glob


class OpeningBook:
    """Collection of pre-recorded opening positions — lazy state construction."""

    def __init__(self, game, dir_path: str):
        self._game = game
        self._serials = []  # raw serialized strings (compact, ~100 bytes each)
        if not dir_path or not os.path.isdir(dir_path):
            return
        # Recursively scan all .txt files in the directory tree
        files = sorted(glob.glob(os.path.join(dir_path, "**", "*.txt"),
                                 recursive=True))
        for fpath in files:
            try:
                with open(fpath) as f:
                    raw = f.read().strip()
                if raw:
                    self._serials.append(raw)
            except Exception:
                pass

    def __len__(self):
        return len(self._serials)

    def __bool__(self):
        return len(self._serials) > 0

    def sample(self, rng):
        """Return a random cloned opening state, or None if empty."""
        if not self._serials:
            return None
        idx = rng.randint(0, len(self._serials))
        raw = self._serials[idx]
        try:
            s = self._game.deserialize_state(raw)
            if not s.is_terminal():
                return s
        except Exception:
            pass
        return None
