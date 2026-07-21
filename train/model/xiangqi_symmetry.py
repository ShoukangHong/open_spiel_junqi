"""Xiangqi symmetry transforms for data augmentation.

4 transforms from 2 generators:
  Mirror (L-R):  flip columns  (col -> 8-col)
  Swap (color + V-flip):  swap Red<->Black pieces, flip rows  (row -> 9-row)

Value [w,d,l] is invariant — always from current-player perspective.
"""

import numpy as np


COLS = 9
ROWS = 10
NUM_ACTIONS = ROWS * COLS * ROWS * COLS  # 8100


class XiangqiSymmetry:
    """4 symmetry transforms for the 10×9 board."""

    num_transforms = 4

    # Plane 15 = move_num / MAX_GAME_LENGTH.
    # Plane 16 = no_capture_count / MAX_NO_CAP_LENGTH.
    MAX_GAME_LENGTH = 400
    MAX_NO_CAP_LENGTH = 40
    MOVE_NUM_PLANE = 15
    NO_CAP_PLANE = 16

    # Randomise move-number when below this many moves.
    MOVE_NUM_RAND_MAX = 200
    # Randomise no-capture when below this many, and only for decisive games.
    NO_CAP_RAND_MAX = 39

    def __init__(self):
        # Precompute action mappings for each transform.
        # inv[k, new_a] = original_a  (same convention as OthelloSymmetry)
        self._inv = np.zeros((4, NUM_ACTIONS), dtype=int)

        for k in range(4):
            mirror = k % 2 == 1      # k=1 or k=3
            swap = k >= 2             # k=2 or k=3

            for from_sq in range(90):
                fr, fc = from_sq // COLS, from_sq % COLS
                if mirror:
                    fc = 8 - fc
                if swap:
                    fr = 9 - fr
                new_from = fr * COLS + fc

                for to_sq in range(90):
                    tr, tc = to_sq // COLS, to_sq % COLS
                    if mirror:
                        tc = 8 - tc
                    if swap:
                        tr = 9 - tr
                    new_to = tr * COLS + tc
                    new_a = new_from * 90 + new_to
                    orig_a = from_sq * 90 + to_sq
                    self._inv[k, new_a] = orig_a

    # ── Public API ──────────────────────────────────────────────────────────

    def augment_batch(self, obs, mask, policy, value=None):
        """Apply random symmetry to each sample in the batch.

        Args:
            obs:    (B, 1530) flat or (B, 17, 10, 9) float32
            mask:   (B, 8100) bool
            policy: (B, 8100) float32
            value:  (B, 3) float32 [w,d,l], optional — invariant under all transforms

        Returns (obs, mask, policy, value_or_none).
        """
        B = obs.shape[0]
        flat = obs.ndim == 2
        choices = np.random.randint(0, self.num_transforms, size=B)

        new_obs = np.empty_like(obs)
        new_mask = np.empty_like(mask)
        new_policy = np.empty_like(policy)
        new_value = None if value is None else value.copy()

        for i, k in enumerate(choices):
            mirror = k % 2 == 1
            swap = k >= 2

            # --- observation ---
            o = obs[i].reshape(17, ROWS, COLS)
            if mirror:
                o = o[:, :, ::-1]                     # flip columns
            if swap:
                o = o[:, ::-1, :]                      # flip rows
                tmp = o[0:7].copy()
                o[0:7] = o[7:14]
                o[7:14] = tmp                           # swap piece planes
                o[14] = 1.0 - o[14]                    # invert turn indicator
            # Randomise move-number plane to an integer count to break
            # spurious phase correlation.
            mn_val = float(o[self.MOVE_NUM_PLANE, 0, 0])
            if mn_val * self.MAX_GAME_LENGTH < self.MOVE_NUM_RAND_MAX:
                new_count = np.random.randint(0, self.MOVE_NUM_RAND_MAX)
                o[self.MOVE_NUM_PLANE] = new_count / float(self.MAX_GAME_LENGTH)

            # For decisive games, reduce no-capture counter to a smaller
            # *integer* count so the model learns compact play in relaxed
            # time contexts.  Drawn games are left untouched.
            nc_val = float(o[self.NO_CAP_PLANE, 0, 0])
            orig_count = int(round(nc_val * self.MAX_NO_CAP_LENGTH))
            if (value is not None and value[i, 1] < 0.3
                    and orig_count < self.NO_CAP_RAND_MAX):
                if orig_count > 1:
                    # max of 2 rolls: long tail toward orig_count, near 0 rare.
                    r1 = np.random.randint(0, orig_count)
                    r2 = np.random.randint(0, orig_count)
                    new_count = max(r1, r2)
                else:
                    new_count = 0
                o[self.NO_CAP_PLANE] = new_count / float(
                    self.MAX_NO_CAP_LENGTH)

            new_obs[i] = o.reshape(-1) if flat else o

            # --- mask & policy ---
            inv = self._inv[k]
            new_mask[i] = mask[i][inv]
            new_policy[i] = policy[i][inv]

        return new_obs, new_mask, new_policy, new_value
