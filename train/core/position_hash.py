"""Position-aware hashing that ignores dynamic observation planes.

Each game registers (static_planes, total_planes) so that the same board
position hashes identically regardless of move counters.
"""
import numpy as np

_GAME_OBS_INFO = {
    "xiangqi": (15, 17),   # skip plane 15 (move#) + 16 (no-cap#)
}


def hash_obs(obs_flat, game_name):
    """Hash the position-relevant prefix of a flattened observation tensor."""
    info = _GAME_OBS_INFO.get(game_name)
    if info is None:
        obs_flat = np.asarray(obs_flat, dtype=np.float32)
    else:
        static_n, total_n = info
        n = int(len(obs_flat) * static_n / total_n)
        obs_flat = np.asarray(obs_flat[:n], dtype=np.float32)
    return hash(obs_flat.tobytes())


def hash_state(state):
    """Hash a pyspiel State — convenience wrapper for hash_obs."""
    obs = np.asarray(state.observation_tensor(), dtype=np.float32)
    return hash_obs(obs, state.get_game().get_type().short_name)
