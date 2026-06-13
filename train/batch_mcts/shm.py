"""Shared-memory buffers for actor↔server communication — no pickle overhead."""

import multiprocessing.shared_memory as _shm
import numpy as np


class ActorShm:
    """Per-actor shared-memory buffer.

    Layout:
      [8B: in_off=8+max_n*(obs_flat*4+mask_flat)][8B: out_cap]
      [input: obs...][mask...][output: policy_logits...][value_logits...]

    The header stores the input section size so both sides agree on offsets
    regardless of how the shm was opened.
    """

    def __init__(self, name, max_states, obs_flat, mask_flat, policy_flat,
                 create=False):
        self._obs_flat = obs_flat
        self._mask_flat = mask_flat
        self._pol_flat = policy_flat
        in_size = max_states * (obs_flat * 4 + mask_flat * 1)
        out_cap = 4 + max_states * (policy_flat * 4 + 3 * 4)
        # 2× safety margin: MCTS may send more than max_states rare cases
        total = 16 + in_size * 2 + out_cap
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
            self._shm.buf[8:16] = np.int64(out_cap).tobytes()
        else:
            self._shm = _shm.SharedMemory(name=name)
        self._in_off = int(np.frombuffer(self._shm.buf[0:8], dtype=np.int64)[0])
        self._out_cap = int(np.frombuffer(self._shm.buf[8:16], dtype=np.int64)[0])
        self._name = name

    @property
    def name(self):
        return self._name

    def close(self):
        try:
            self._shm.close()
        except BufferError:
            pass  # numpy arrays still hold buffer views, safe to ignore

    def unlink(self):
        try:
            self._shm.unlink()
        except Exception:
            pass

    # ── Actor side ──────────────────────────────────────────────────────────

    def _write_bytes(self, offset, data):
        """Fallback: copy bytes via intermediate numpy array (Windows-safe)."""
        buf = np.ndarray(len(self._shm.buf), dtype=np.byte, buffer=self._shm.buf)
        buf[offset:offset + len(data)] = np.frombuffer(data, dtype=np.uint8)

    def _has_direct_view(self):
        """True on Linux where np.ndarray(buffer=...) works reliably."""
        import sys
        return sys.platform != "win32"

    def write_input(self, obs, mask):
        n = obs.shape[0]
        self._write_bytes(0, np.int32(n).tobytes())
        off = self._in_off
        if self._has_direct_view():
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

    def write_output(self, policy_logits, value_logits):
        """Pad policy to pol_flat, write to output area."""
        n = policy_logits.shape[0]
        pad = self._pol_flat - policy_logits.shape[1]
        end = self._in_off + self._out_cap
        total = n * (self._pol_flat * 4 + 3 * 4)
        off = end - 4 - total
        if pad > 0:
            pl = np.pad(policy_logits.astype(np.float32),
                        ((0, 0), (0, pad)))
        else:
            pl = policy_logits
        if self._has_direct_view():
            dst = np.ndarray(pl.shape, dtype=np.float32,
                             buffer=self._shm.buf, offset=off)
            np.copyto(dst, pl)
            off += n * self._pol_flat * 4
            dst = np.ndarray(value_logits.shape, dtype=np.float32,
                             buffer=self._shm.buf, offset=off)
            np.copyto(dst, value_logits)
        else:
            self._write_bytes(off, np.ascontiguousarray(pl).tobytes())
            off += n * self._pol_flat * 4
            self._write_bytes(off,
                              np.ascontiguousarray(value_logits).tobytes())
        self._write_bytes(end - 4, np.int32(n).tobytes())

    def read_output(self):
        """Read n from very end; read max-sized block before it."""
        end = self._in_off + self._out_cap
        n = int(np.frombuffer(self._shm.buf[end - 4:end], dtype=np.int32)[0])
        if n <= 0:
            return None
        off = end - 4 - n * (self._pol_flat * 4 + 3 * 4)
        data = np.frombuffer(self._shm.buf, dtype=np.float32,
                             count=n * (self._pol_flat + 3), offset=off)
        return (data[:n * self._pol_flat].reshape(n, self._pol_flat),
                data[n * self._pol_flat:].reshape(n, 3))
        return (policy_logits.reshape(n, self._pol_flat),
                value_logits.reshape(n, 3))
