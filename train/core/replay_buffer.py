"""FIFO ring buffer for AlphaZero (obs, mask, policy, value, tag) samples."""

import numpy as np
from train.model.othello_resnet import TrainInput


class ReplayBuffer:
    """Fixed-size FIFO ring buffer for training tuples."""

    def __init__(self, max_size: int, value_dim: int = 1):
        self._max_size = max_size
        self._value_dim = value_dim
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
            v_shape = (self._max_size,) if self._value_dim == 1 else \
                       (self._max_size, self._value_dim)
            self._values = np.empty(v_shape, dtype=np.float32)
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

        # Detect value shape from saved data (handles shape changes)
        saved_values = data["values"]
        if saved_values.ndim > 1:
            self._value_dim = saved_values.shape[1]
        v_shape = (self._max_size,) if saved_values.ndim == 1 else \
                   (self._max_size, saved_values.shape[1])
        self._values = np.empty(v_shape, dtype=np.float32)
        self._tags = np.empty((self._max_size,), dtype=object)

        self._obs[:self._size] = data["obs"]
        self._masks[:self._size] = data["masks"]
        self._policies[:self._size] = data["policies"]
        self._values[:self._size] = data["values"]
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
