"""Game-specific symmetry transforms for data augmentation.

Each symmetry transform maps an Othello board + policy/mask to an
equivalent rotated/flipped version.  The value target is invariant.
"""

import numpy as np


class OthelloSymmetry:
    """D4 group × color swap = 16 transforms for the 8×8 board.

    8 spatial transforms (4 rotations × mirror), each with optional
    color flip (swap black/white pieces + invert player-to-move).
    """

    num_transforms = 16

    def __init__(self):
        # (16, 65): inv[k, new_a] = original action
        self._inv = np.zeros((16, 65), dtype=int)
        self._inv[:, 64] = 64  # pass stays pass

        for k in range(8):
            rot = k % 4
            flip = k // 4
            for a in range(64):
                r, c = a // 8, a % 8
                if flip:
                    c = 7 - c
                for _ in range(rot):
                    r, c = 7 - c, r      # 90° CCW (matches np.rot90 default)
                self._inv[k, r * 8 + c] = a

        # Color-flip transforms (k=8..15): same spatial mapping, same _inv
        self._inv[8:16] = self._inv[0:8]

    # ── Public API ──────────────────────────────────────────────────────────

    def augment_batch(self, obs, mask, policy, value=None):
        """Apply random symmetries to a batch.

        Args:
            obs:    (B, 256) flat or (B, 4, 8, 8) float32
            mask:   (B, 65) bool
            policy: (B, 65) float32
            value:  (B, 3) float32 [w,d,l], optional

        Returns (obs, mask, policy, value_or_none).
        WDL is unchanged by color flip (turn indicator in obs compensates).
        """
        B = obs.shape[0]
        flat = obs.ndim == 2
        choices = np.random.randint(0, self.num_transforms, size=B)

        new_obs = np.empty_like(obs)
        new_mask = np.empty_like(mask)
        new_policy = np.empty_like(policy)
        new_value = None if value is None else value.copy()

        for i, k in enumerate(choices):
            rot, flip = k % 4, k // 4
            swapped = k >= 8

            # --- observation ---
            o = obs[i].reshape(4, 8, 8)
            if flip:
                o = o[:, :, ::-1]
            if rot:
                o = np.rot90(o, rot, axes=(1, 2))
            if swapped:
                o[[1, 2]] = o[[2, 1]]          # swap black ↔ white
                o[3] = 1.0 - o[3]              # invert turn indicator
            new_obs[i] = o.reshape(-1) if flat else o

            # --- mask & policy ---
            inv = self._inv[k % 8]
            new_mask[i] = mask[i][inv]
            new_policy[i] = policy[i][inv]

        return new_obs, new_mask, new_policy, new_value
