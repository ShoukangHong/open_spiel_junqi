"""FIFO ring buffer for AlphaZero (obs, mask, policy, value, tag) samples."""

import numpy as np
from train.model.othello_resnet import TrainInput


class ReplayBuffer:
    """Fixed-size FIFO ring buffer for training tuples."""

    def __init__(self, max_size: int):
        self._max_size = max_size
        self._obs = None
        self._masks = None
        self._policies = None
        self._values = None
        self._tags = None
        self._index = 0
        self._size = 0

    def append(self, obs: np.ndarray, mask: np.ndarray,
               policy: np.ndarray, value, tag: str = ""):
        if self._obs is None:
            self._obs = np.empty((self._max_size, *obs.shape), dtype=np.float32)
            self._masks = np.empty((self._max_size, *mask.shape), dtype=bool)
            self._policies = np.empty((self._max_size, *policy.shape),
                                       dtype=np.float32)
            self._values = np.empty((self._max_size, 3), dtype=np.float32)
            self._tags = np.empty((self._max_size,), dtype=object)

        idx = self._index % self._max_size
        self._obs[idx] = obs.astype(np.float32)
        self._masks[idx] = mask
        self._policies[idx] = policy.astype(np.float32)
        self._values[idx] = np.asarray(value, dtype=np.float32)
        self._tags[idx] = tag
        self._index += 1
        self._size = min(self._size + 1, self._max_size)

    def sample(self, n: int) -> TrainInput:
        indices = np.random.randint(0, self._size, size=n)
        return TrainInput(
            observation=self._obs[indices],
            legals_mask=self._masks[indices],
            policy=self._policies[indices],
            value=self._values[indices],
        )

    def tag_counts(self) -> dict:
        if self._tags is None:
            return {}
        tags = self._tags[:self._size]
        valid = [str(t) for t in tags if t is not None]
        if not valid:
            return {}
        unique, counts = np.unique(valid, return_counts=True)
        return {str(k): int(v) for k, v in zip(unique, counts)}

    def save(self, filepath: str):
        if self._obs is None:
            return
        tags_arr = np.array(self._tags[:self._size], dtype=str)
        np.savez_compressed(
            filepath,
            obs=self._obs[:self._size],
            masks=self._masks[:self._size],
            policies=self._policies[:self._size],
            values=self._values[:self._size],
            tags=tags_arr,
            index=self._index,
            size=self._size,
        )

    def load(self, filepath: str):
        data = np.load(filepath)
        self._size = int(data["size"])
        self._index = int(data["index"])
        self._max_size = max(self._max_size, self._size)
        obs_shape = data["obs"].shape[1:]
        mask_shape = data["masks"].shape[1:]
        policy_shape = data["policies"].shape[1:]
        self._obs = np.empty((self._max_size, *obs_shape), dtype=np.float32)
        self._masks = np.empty((self._max_size, *mask_shape), dtype=bool)
        self._policies = np.empty((self._max_size, *policy_shape), dtype=np.float32)

        saved_values = data["values"]
        if saved_values.ndim == 1:
            raise ValueError(
                "Old scalar buffer format detected.  Convert with:\n"
                "  python train/core/replay_buffer.py <old.npz>")
        self._values = np.empty((self._max_size, 3), dtype=np.float32)
        self._tags = np.empty((self._max_size,), dtype=object)

        self._obs[:self._size] = data["obs"]
        self._masks[:self._size] = data["masks"]
        self._policies[:self._size] = data["policies"]
        self._values[:self._size] = saved_values[:self._size]
        if "tags" in data:
            self._tags[:self._size] = data["tags"]

    def __len__(self) -> int:
        return self._size

    @property
    def total_seen(self) -> int:
        return self._index

    @property
    def unique_states(self) -> int:
        if self._obs is None or self._size == 0:
            return 0
        seen = set()
        for i in range(self._size):
            seen.add(self._obs[i].tobytes())
        return len(seen)


# ── Buffer converter (old scalar → new WDL) ──────────────────────────────────

def _convert_buffer(src_path: str, dst_path: str = None):
    """Convert old scalar buffer to WDL format.  Writes in-place if no dst."""
    import os as _os
    data = np.load(src_path)
    vals = data["values"]
    if vals.ndim > 1:
        data.close()
        print(f"[convert] Already WDL format ({vals.shape[1]}-dim), skipping.")
        return

    wdl = np.zeros((vals.shape[0], 3), dtype=np.float32)
    v = vals.astype(np.float32)
    wdl[:, 0] = np.maximum(v, 0)
    wdl[:, 1] = 1.0 - np.abs(v)
    wdl[:, 2] = np.maximum(-v, 0)

    obs = data["obs"].copy()
    masks = data["masks"].copy()
    policies = data["policies"].copy()
    tags = data.get("tags", np.array([""] * vals.shape[0]))
    idx = int(data.get("index", vals.shape[0]))
    sz = int(data.get("size", vals.shape[0]))
    data.close()

    out = dst_path or src_path
    tmp = out + ".converting.npz"
    np.savez_compressed(tmp, obs=obs, masks=masks, policies=policies,
                        values=wdl, tags=tags, index=idx, size=sz)
    _os.replace(tmp, out)
    print(f"[convert] {vals.shape[0]} states: scalar → WDL → {out}")


if __name__ == "__main__":
    import glob
    target = r"C:\Users\shouk\othello_train\cloud_wdl_w\buffer-checkpoint-240.npz"
    paths = sorted(glob.glob(f"{target}/buffer-checkpoint-*.npz")) if __import__("os").path.isdir(target) else [target]
    for p in paths:
        _convert_buffer(p)
