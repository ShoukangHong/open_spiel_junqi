"""Offline training from existing replay buffer DBs.

Replaces self-play data generation with pre-recorded states from buffer_*.db
files.  Uses a sliding window: as training progresses, older buffer states
are evicted and newer ones loaded, mimicking the online FIFO ring.

Usage:
    python experiment/buffer_train.py                          # auto-detect
    python experiment/buffer_train.py --config path/to/config.json
    python experiment/buffer_train.py --config config.json --fresh
"""

import argparse
import glob as _glob
import logging
import math
import os
import re
import sqlite3
import sys
import time
import zlib

_sys_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _sys_root not in sys.path:
    sys.path.insert(0, _sys_root)

import numpy as np
import pyspiel
import torch
torch.set_num_threads(1)

from train.core.base_config import BaseTrainConfig
from train.core.checkpoint import find_latest_checkpoint
from train.core.checkpoint import find_latest_checkpoint
from train.core.replay_buffer import _unpack
from train.core.train_utils import setup_config_and_logging
from train.core.types import TrainInput, Losses
from train.games.xiangqi.config import XiangqiTrainConfig
from train.model.xiangqi_symmetry import XiangqiSymmetry


def _load_db_rows(db_path: str, limit: int = 0, offset: int = 0,
                  row_count_out: list = None):
    """Read rows from a buffer DB.  Returns (lists, is_exhausted)."""
    conn = sqlite3.connect(db_path)
    sql = "SELECT obs, mask, policy, value, tag, step FROM states ORDER BY id"
    if limit > 0:
        sql += f" LIMIT {limit} OFFSET {offset}"
    cur = conn.execute(sql)
    rows = cur.fetchall()
    # Check total row count so we know if more rows remain
    remaining = 0
    if row_count_out is not None:
        total_cur = conn.execute("SELECT COUNT(*) FROM states")
        total = total_cur.fetchone()[0]
        row_count_out[0] = total
        remaining = total - offset - len(rows)
    conn.close()
    if not rows:
        return [], [], [], [], [], [], remaining <= 0
    obs_list, mask_list, pol_list, val_list, tag_list, step_list = [], [], [], [], [], []
    for obs_b, mask_b, pol_b, val_b, tag, step in rows:
        obs_list.append(_unpack(obs_b, np.float32))
        mask_list.append(_unpack(mask_b, bool))
        pol_list.append(_unpack(pol_b, np.float32))
        val_list.append(_unpack(val_b, np.float32))
        tag_list.append(tag)
        step_list.append(step)
    return obs_list, mask_list, pol_list, val_list, tag_list, step_list, remaining <= 0


def run_buffer_training(config_class=XiangqiTrainConfig,
                        config_path: str = None,
                        output_path: str = None,
                        fresh: bool = False):
    """Run offline training using buffer DBs as data source."""

    cfg = setup_config_and_logging(config_path, config_class, fresh,
                                   output_dir=output_path)
    _log = logging.info

    # ── Create model ─────────────────────────────────────────────────
    game = pyspiel.load_game(cfg.game)
    model = _build_model(game, cfg)
    sym = XiangqiSymmetry() if getattr(cfg, 'symmetry', 1) > 1 else None
    start_step = 0
    if not fresh:
        start_step = find_latest_checkpoint(cfg.path)
        if start_step > 0:
            model.load_checkpoint(start_step)
    _log(f"[buffer_train] params={model.num_trainable_variables}  "
         f"lr={cfg.learning_rate:.0e}  start_step={start_step}")

    # ── Enumerate buffer DB files in chronological order ────────────────
    db_dir = cfg.path
    all_dbs = sorted(
        [f for f in _glob.glob(os.path.join(db_dir, "buffer_*.db"))
         if re.search(r'buffer_(\d+)\.db', f)],
        key=lambda f: int(re.search(r'buffer_(\d+)\.db', f).group(1)))
    _log(f"[buffer_train] Found {len(all_dbs)} buffer DBs in {db_dir}")

    # Latest buffer.db (no step suffix)
    latest_db = os.path.join(db_dir, "buffer.db")

    # ── Compute how many states to load per step ────────────────────────
    states_per_step = max(
        int(cfg.replay_buffer_size * cfg.buffer_sampling_frac),
        cfg.train_batch_size)
    max_mem = cfg.replay_buffer_size

    # ── In-memory storage (flat arrays) ─────────────────────────────────
    in_mem = dict(obs=[], mask=[], policy=[], value=[], tag=[], step=[])
    mem_count = 0
    db_idx = 0        # index into all_dbs
    db_offset = 0      # offset within the current DB

    def _append_to_mem(obs, mask, policy, value, tag, s):
        nonlocal mem_count
        in_mem["obs"].append(obs.astype(np.float32))
        in_mem["mask"].append(mask)
        in_mem["policy"].append(policy.astype(np.float32))
        in_mem["value"].append(np.asarray(value, dtype=np.float32))
        in_mem["tag"].append(tag)
        in_mem["step"].append(s)
        mem_count += 1

    def _evict_oldest(n: int):
        """Remove the oldest *n* entries from in-memory storage."""
        nonlocal mem_count
        n = min(n, mem_count)
        for k in in_mem:
            in_mem[k] = in_mem[k][n:]
        mem_count -= n

    def _load_next_db(limit: int = 0):
        """Load up to *limit* states from buffer DBs, resuming partial reads."""
        nonlocal db_idx, db_offset
        loaded = 0
        while loaded < limit and db_idx < len(all_dbs):
            path = all_dbs[db_idx]
            want = limit - loaded if limit > 0 else 0
            total_info = [0]
            obs_l, mask_l, pol_l, val_l, tag_l, step_l, exhausted = \
                _load_db_rows(path, limit=want, offset=db_offset,
                              row_count_out=total_info)
            for obs, mask, policy, value, tag, s in zip(
                    obs_l, mask_l, pol_l, val_l, tag_l, step_l):
                _append_to_mem(obs, mask, policy, value, tag, s)
            n = len(obs_l)
            loaded += n
            db_offset += n
            _log(f"[buffer_train] {os.path.basename(path)}"
                 f" offset={db_offset-n}+{n}"
                 f" total_rows={total_info[0]}"
                 f"  (mem: {mem_count}/{max_mem})")
            if exhausted:
                db_idx += 1
                db_offset = 0
            if n == 0:
                break
        return loaded

    def _sample_batch(n: int):
        """Sample *n* random states from in-memory storage."""
        indices = np.random.randint(0, mem_count, size=n)
        return TrainInput(
            observation=np.stack([in_mem["obs"][i] for i in indices]),
            legals_mask=np.stack([in_mem["mask"][i] for i in indices]),
            policy=np.stack([in_mem["policy"][i] for i in indices]),
            value=np.stack([in_mem["value"][i] for i in indices]),
        )

    # ── Initial fill ────────────────────────────────────────────────────
    while mem_count < max_mem and db_idx < len(all_dbs):
        _load_next_db(limit=max_mem - mem_count)
    _log(f"[buffer_train] Initial fill: {mem_count} states in memory")

    # ── Training loop ───────────────────────────────────────────────────
    for step in range(start_step + 1, cfg.max_steps + 1):
        t0 = time.time()

        # ── Training ─────────────────────────────────────────────────
        samples = max(int(min(mem_count, max_mem) * cfg.buffer_sampling_frac),
                      cfg.train_batch_size)
        n_updates = samples * cfg.symmetry // cfg.train_batch_size
        losses_list = []
        entropies = []
        outcomes = {"p0": 0, "p1": 0, "draw": 0}

        for _ in range(n_updates):
            batch = _sample_batch(cfg.train_batch_size)
            # Symmetry augmentation
            obs, mask, policy, value = sym.augment_batch(
                batch.observation, batch.legals_mask,
                batch.policy, batch.value)
            batch = TrainInput(observation=obs, legals_mask=mask,
                               policy=policy, value=value)
            model._entropy_weight = cfg.entropy_weight
            loss = model.update(batch)
            losses_list.append(loss)

            # Entropy of target policy
            p = batch.policy
            p = np.where(p > 0, p, 1.0)
            entropies.append(float(np.mean(-np.sum(p * np.log(p), axis=-1))))

            # Outcome from target value
            v = batch.value
            w_idx = np.argmax(v, axis=-1)
            outcomes["p0"] += int(np.sum(w_idx == 0))
            outcomes["p1"] += int(np.sum(w_idx == 1))
            outcomes["draw"] += int(np.sum(w_idx == 2))

        train_time = time.time() - t0

        # ── Logging ──────────────────────────────────────────────────
        if losses_list:
            n = len(losses_list)
            avg_loss = Losses(
                policy=sum(l.policy for l in losses_list) / n,
                value=sum(l.value for l in losses_list) / n,
                l2=sum(l.l2 for l in losses_list) / n,
                v_kl=sum(l.v_kl for l in losses_list) / n,
                top1=sum(l.top1 for l in losses_list) / n,
                p_kl=sum(l.p_kl for l in losses_list) / n,
                grad_rel=sum(l.grad_rel for l in losses_list) / n,
                update_rel=sum(l.update_rel for l in losses_list) / n,
            )
            h = min(4, n)
            p_kl_head = sum(l.p_kl for l in losses_list[:h]) / h
            mid = max(0, n // 2 - 2)
            p_kl_mid = sum(l.p_kl for l in losses_list[mid:mid + h]) / max(1, min(h, n - mid))
            p_kl_tail = sum(l.p_kl for l in losses_list[-h:]) / h
            v_kl_head = sum(l.v_kl for l in losses_list[:h]) / h
            v_kl_mid = sum(l.v_kl for l in losses_list[mid:mid + h]) / max(1, min(h, n - mid))
            v_kl_tail = sum(l.v_kl for l in losses_list[-h:]) / h
            avg_ent = sum(entropies) / len(entropies)
            g = sum(outcomes.values()) or 1
        else:
            p_kl_head = p_kl_mid = p_kl_tail = 0.0
            v_kl_head = v_kl_mid = v_kl_tail = 0.0
            avg_loss = None
            avg_ent = 0.0
            g = 1

        _log(f"[step {step:4d}/{cfg.max_steps}] "
             f"updates={n_updates}  mem={mem_count}/{max_mem}  "
             f"db={db_idx}/{len(all_dbs)}  train={train_time:.1f}s")
        if avg_loss is not None:
            _log(f"               loss={avg_loss}"
                 f"  entropy={avg_ent:.3f}  |"
                 f"  pkl_h={p_kl_head:.4f} pkl_m={p_kl_mid:.4f}"
                 f"  pkl_t={p_kl_tail:.4f}"
                 f"  vkl_h={v_kl_head:.4f} vkl_m={v_kl_mid:.4f}"
                 f"  vkl_t={v_kl_tail:.4f}"
                 f"  p0={outcomes['p0']/g:.1%}"
                 f"  p1={outcomes['p1']/g:.1%}"
                 f"  draw={outcomes['draw']/g:.1%}")

        # ── Load next buffer(s) — sliding window ──────────────────────
        _load_next_db(limit=states_per_step)
        # Evict oldest to stay within max_mem
        if mem_count > max_mem:
            _evict_oldest(mem_count - max_mem)

        # ── Checkpoint ────────────────────────────────────────────────
        if step % cfg.checkpoint_freq == 0:
            ckpt_path = model.save_checkpoint(step)
            _log(f"  [checkpoint] Saved {ckpt_path}")

    # ── Final: load buffer.db (latest) and train a few more steps ──────
    if os.path.exists(latest_db) and latest_db not in all_dbs:
        obs_l, mask_l, pol_l, val_l, tag_l, step_l = _load_db_rows(latest_db)
        for obs, mask, policy, value, tag, s in zip(
                obs_l, mask_l, pol_l, val_l, tag_l, step_l):
            _append_to_mem(obs, mask, policy, value, tag, s)
        _log(f"[buffer_train] Loaded buffer.db: {len(obs_l)} states")

        # Train on the latest data
        extra_steps = 5
        for step in range(cfg.max_steps + 1, cfg.max_steps + extra_steps + 1):
            samples = max(int(min(mem_count, max_mem) * cfg.buffer_sampling_frac),
                          cfg.train_batch_size)
            n_updates = samples * cfg.symmetry // cfg.train_batch_size
            for _ in range(n_updates):
                batch = _sample_batch(cfg.train_batch_size)
                obs, mask, policy, value = sym.augment_batch(
                    batch.observation, batch.legals_mask,
                    batch.policy, batch.value)
                batch = TrainInput(observation=obs, legals_mask=mask,
                                   policy=policy, value=value)
                model._entropy_weight = cfg.entropy_weight
                model.update(batch)
            _log(f"[step {step:4d}] buffer.db final training  "
                 f"updates={n_updates}  mem={mem_count}")
            if step % cfg.checkpoint_freq == 0:
                model.save_checkpoint(step)

    # Final save
    model.save_checkpoint(cfg.max_steps + extra_steps)
    _log(f"\n[buffer_train] Done. Final checkpoint: {cfg.max_steps + extra_steps}")


# Minimal stubs to reuse init_training
def _build_model(game, cfg):
    from train.core.model_builder import build_xiangqi_model
    return build_xiangqi_model(game, {
        "nn_width": cfg.nn_width, "nn_depth": cfg.nn_depth,
        "device": cfg.device, "path": cfg.path,
    })


def main():
    parser = argparse.ArgumentParser(
        description="Offline training from buffer DB files")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (overrides config.path)")
    parser.add_argument("--fresh", action="store_true",
                        help="Start from scratch, ignoring existing checkpoints")
    args = parser.parse_args()
    run_buffer_training(config_path=args.config, output_path=args.output,
                        fresh=args.fresh)


if __name__ == "__main__":
    main()
