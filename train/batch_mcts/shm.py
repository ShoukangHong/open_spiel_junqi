"""Shared-memory buffers for actor↔server communication — no pickle overhead.

ActorShm: per-actor, input-only (actor → server).
ServerShm: single output buffer per GPU (server → all actors).

Layout — ActorShm:
  [8B: in_off][8B: reserved]
  [input: obs...][mask...]  2× safety margin

Layout — ServerShm:
  [8B: total_n]
  [policy_logits: max_batch * pol_flat * 4]
  [value_logits: max_batch * 3 * 4]
"""

import multiprocessing.shared_memory as _shm
import numpy as np


def _has_direct_view():
    """True on Linux where np.ndarray(buffer=...) works reliably."""
    import sys
    return sys.platform != "win32"


def _write_bytes(shm, offset, data):
    """Fallback: copy bytes via intermediate array (Windows-safe)."""
    buf = np.ndarray(len(shm.buf), dtype=np.byte, buffer=shm.buf)
    buf[offset:offset + len(data)] = np.frombuffer(data, dtype=np.uint8)


# ── Per-actor input buffer ────────────────────────────────────────────────────

class ActorShm:
    """Per-actor shared-memory buffer — input only (actor → server).

    The actor writes obs (and optionally mask) then sends a short queue
    message.  The server reads directly from this buffer.
    """

    def __init__(self, name, max_states, obs_flat, mask_flat, create=False):
        self._obs_flat = obs_flat
        self._mask_flat = mask_flat
        in_size = max_states * (obs_flat * 4 + mask_flat * 1)
        total = 16 + in_size * 2
        if create:
            try:
                self._shm = _shm.SharedMemory(name=name, create=True, size=total)
            except FileExistsError:
                try:
                    old = _shm.SharedMemory(name=name)
                    old.close()
                    old.unlink()
                except Exception:
                    pass
                self._shm = _shm.SharedMemory(name=name, create=True, size=total)
            self._shm.buf[:total] = b'\x00' * total
            self._shm.buf[0:8] = np.int64(16 + in_size).tobytes()
            self._shm.buf[8:16] = np.int64(0).tobytes()
        else:
            self._shm = _shm.SharedMemory(name=name)
        self._in_off = int(np.frombuffer(self._shm.buf[0:8], dtype=np.int64)[0])
        self._name = name

    @property
    def name(self):
        return self._name

    def close(self):
        try:
            self._shm.close()
        except BufferError:
            pass

    def unlink(self):
        try:
            self._shm.unlink()
        except Exception:
            pass

    # ── Actor side ──────────────────────────────────────────────────────────

    def write_input_obs(self, obs):
        """Server only needs obs — skip mask for zero-copy efficiency."""
        n = obs.shape[0]
        self._write_bytes(0, np.int32(n).tobytes())
        off = self._in_off
        if _has_direct_view():
            dst = np.ndarray(obs.shape, dtype=np.float32,
                             buffer=self._shm.buf, offset=off)
            np.copyto(dst, obs)
        else:
            self._write_bytes(off, obs.astype(np.float32).tobytes(order='C'))

    def write_input(self, obs, mask):
        n = obs.shape[0]
        self._write_bytes(0, np.int32(n).tobytes())
        off = self._in_off
        if _has_direct_view():
            dst = np.ndarray(obs.shape, dtype=np.float32,
                             buffer=self._shm.buf, offset=off)
            np.copyto(dst, obs)
            off += n * self._obs_flat * 4
            dst = np.ndarray(mask.shape, dtype=np.bool_,
                             buffer=self._shm.buf, offset=off)
            np.copyto(dst, mask)
        else:
            self._write_bytes(off, obs.astype(np.float32).tobytes(order='C'))
            off += n * self._obs_flat * 4
            self._write_bytes(off, mask.astype(np.bool_).tobytes(order='C'))

    # ── Server side ─────────────────────────────────────────────────────────

    def read_input(self):
        n = int(np.frombuffer(self._shm.buf[0:4], dtype=np.int32)[0])
        if n <= 0:
            return None
        off = self._in_off
        size = n * self._obs_flat
        obs = np.frombuffer(self._shm.buf, dtype=np.float32, count=size, offset=off)
        off += size * 4
        size = n * self._mask_flat
        mask = np.frombuffer(self._shm.buf, dtype=np.bool_, count=size, offset=off)
        return obs.reshape(n, self._obs_flat), mask.reshape(n, self._mask_flat)

    def read_input_obs(self):
        """Server-side: read only obs, skip mask (not needed for forward)."""
        n = int(np.frombuffer(self._shm.buf[0:4], dtype=np.int32)[0])
        if n <= 0:
            return None
        off = self._in_off
        size = n * self._obs_flat
        return np.frombuffer(self._shm.buf, dtype=np.float32,
                             count=size, offset=off).reshape(n, self._obs_flat)

    def _write_bytes(self, offset, data):
        _write_bytes(self._shm, offset, data)


# ── Shared output buffer (one per GPU server) ─────────────────────────────────

class ServerShm:
    """Single output buffer shared by all actors on one GPU.

    Server writes inference results once; each actor reads its slice
    using the (offset, n) it received in the queue notification.
    """

    def __init__(self, name, max_states, pol_flat, create=False):
        self._pol_flat = pol_flat
        out_size = max_states * (pol_flat * 4 + 3 * 4)
        total = 8 + out_size
        self._name = name
        if create:
            try:
                self._shm = _shm.SharedMemory(name=name, create=True, size=total)
            except FileExistsError:
                try:
                    old = _shm.SharedMemory(name=name)
                    old.close()
                    old.unlink()
                except Exception:
                    pass
                self._shm = _shm.SharedMemory(name=name, create=True, size=total)
        else:
            self._shm = _shm.SharedMemory(name=name)
        if create:
            self._shm.buf[0:8] = np.int64(0).tobytes()

    @property
    def name(self):
        return self._name

    def close(self):
        try:
            self._shm.close()
        except BufferError:
            pass

    def unlink(self):
        try:
            self._shm.unlink()
        except Exception:
            pass

    # ── Server side ─────────────────────────────────────────────────────────

    def write_batch(self, policy_logits, value_logits):
        """Write a full batch of inference results (one forward call)."""
        n = policy_logits.shape[0]
        self._shm.buf[0:8] = np.int64(n).tobytes()
        off = 8
        if _has_direct_view():
            dst = np.ndarray(policy_logits.shape, dtype=np.float32,
                             buffer=self._shm.buf, offset=off)
            np.copyto(dst, policy_logits)
            off += n * self._pol_flat * 4
            dst = np.ndarray(value_logits.shape, dtype=np.float32,
                             buffer=self._shm.buf, offset=off)
            np.copyto(dst, value_logits)
        else:
            _write_bytes(self._shm, off,
                         np.ascontiguousarray(policy_logits).tobytes())
            off += n * self._pol_flat * 4
            _write_bytes(self._shm, off,
                         np.ascontiguousarray(value_logits).tobytes())

    # ── Actor side ──────────────────────────────────────────────────────────

    def read_slice(self, offset, n):
        """Read n states starting at *offset* within the batch."""
        total_n = int(np.frombuffer(self._shm.buf[0:8], dtype=np.int64)[0])
        if total_n <= 0 or n <= 0:
            return np.empty((0, self._pol_flat), dtype=np.float32), \
                   np.empty((0, 3), dtype=np.float32)
        pol_start = 8 + offset * self._pol_flat * 4
        pol = np.frombuffer(self._shm.buf, dtype=np.float32,
                            count=n * self._pol_flat, offset=pol_start)
        val_start = 8 + total_n * self._pol_flat * 4 + offset * 3 * 4
        val = np.frombuffer(self._shm.buf, dtype=np.float32,
                            count=n * 3, offset=val_start)
        return pol.reshape(n, self._pol_flat), val.reshape(n, 3)
