"""ActorShm round-trip tests with Xiangqi dimensions."""

import numpy as np
from train.batch_mcts.shm import ActorShm


def test_shm_xiangqi():
    shm = ActorShm("test_xq", 16, 1530, 8100, create=True)
    try:
        for n in [1, 3, 16]:
            obs = np.random.randn(n, 1530).astype(np.float32)
            mask = np.random.rand(n, 8100) > 0.5
            shm.write_input(obs, mask)
            o2, m2 = shm.read_input()
            assert o2.shape == (n, 1530)
            assert m2.shape == (n, 8100)
            np.testing.assert_array_equal(obs, o2)
            np.testing.assert_array_equal(mask, m2)
    finally:
        _safe_cleanup(shm)


def _safe_cleanup(shm):
    import gc
    gc.collect()
    try:
        shm.close()
    except Exception:
        pass
    try:
        shm.unlink()
    except Exception:
        pass
