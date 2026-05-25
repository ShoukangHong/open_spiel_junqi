"""Game-specific symmetry transforms for data augmentation.

Each symmetry transform maps an Othello board + policy/mask to an
equivalent rotated/flipped version.  The value target is invariant.
"""

import numpy as np


class OthelloSymmetry:
    """D4 group: 4 rotations × mirror = 8 transforms for the 8×8 board."""

    num_transforms = 8

    def __init__(self):
        # (8, 65): action_map[k, a] = new action after transform k
        # (8, 65): inv[k, new_a] = original action
        self._inv = np.zeros((8, 65), dtype=int)
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

    # ── Public API ──────────────────────────────────────────────────────────

    def augment_batch(self, obs, mask, policy):
        """Apply random symmetries to a batch.

        Args:
            obs:    (B, 256) flat or (B, 4, 8, 8) float32
            mask:   (B, 65) bool
            policy: (B, 65) float32

        Returns (obs, mask, policy) — same shapes as inputs.
        Value is invariant (caller handles it).
        """
        B = obs.shape[0]
        flat = obs.ndim == 2
        choices = np.random.randint(0, self.num_transforms, size=B)

        new_obs = np.empty_like(obs)
        new_mask = np.empty_like(mask)
        new_policy = np.empty_like(policy)

        for i, k in enumerate(choices):
            rot, flip = k % 4, k // 4

            # --- observation (→ 4,8,8 → transform → back) ---
            o = obs[i].reshape(4, 8, 8)
            if flip:
                o = o[:, :, ::-1]
            if rot:
                o = np.rot90(o, rot, axes=(1, 2))
            new_obs[i] = o.reshape(-1) if flat else o

            # --- mask & policy (65,) ---
            inv = self._inv[k]
            new_mask[i] = mask[i][inv]
            new_policy[i] = policy[i][inv]

        return new_obs, new_mask, new_policy
