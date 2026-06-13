"""ActorShm round-trip tests with Othello dimensions."""

import numpy as np
from train.batch_mcts.shm import ActorShm


def test_shm_othello():
    shm = ActorShm("test_oth", 16, 256, 65, 65, create=True)
    try:
        for n in [1, 3, 16]:
            obs = np.random.randn(n, 256).astype(np.float32)
            mask = np.random.rand(n, 65) > 0.5
            shm.write_input(obs, mask)
            o2, m2 = shm.read_input()
            assert o2.shape == (n, 256)
            assert m2.shape == (n, 65)
            np.testing.assert_array_equal(obs, o2)
            np.testing.assert_array_equal(mask, m2)

            pl = np.random.randn(n, 65).astype(np.float32)
            vl = np.random.randn(n, 3).astype(np.float32)
            shm.write_output(pl, vl)
            p2, v2 = shm.read_output()
            assert p2.shape == (n, 65)
            assert v2.shape == (n, 3)
            np.testing.assert_array_equal(pl, p2)
            np.testing.assert_array_equal(vl, v2)
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
