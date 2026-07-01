"""Fit model to a single buffer DB — diagnostic for policy learning.

Usage:
    python -m experiment.fit_single_buffer
    python -m experiment.fit_single_buffer --policy-lr 0.0001
"""

import argparse
import math
import os
import sqlite3
import time
import zlib

import numpy as np
import torch
import torch.nn.functional as F

from train.core.model_builder import build_xiangqi_model
from train.core.replay_buffer import _unpack
from train.core.types import TrainInput, Losses
from train.model.xiangqi_symmetry import XiangqiSymmetry
import json, pyspiel

# ── Config ───────────────────────────────────────────────────────────────────

DB_PATH = r"C:\Users\shouk\xiangqi_train\cloud_new\buffer_8693824.db"
CKPT_DIR = r"C:\Users\shouk\xiangqi_train\cloud_new"
CKPT_STEP = 180
CONFIG_PATH = os.path.join(CKPT_DIR, "train_config.json")
NUM_BATCHES = 2000
LOG_EVERY = 50
SUBSET_SIZE = 1000_000  # sample this many rows from DB once, then train in-memory


def load_model_and_config():
    with open(CONFIG_PATH) as f:
        cfg_dict = json.load(f)
    game = pyspiel.load_game("xiangqi")
    model_cfg = {
        "nn_width": cfg_dict["nn_width"],
        "nn_depth": cfg_dict["nn_depth"],
        "device": "cuda",
        "path": CKPT_DIR,
    }
    model = build_xiangqi_model(game, model_cfg)
    model.load_checkpoint(CKPT_STEP)
    return model, cfg_dict


def load_id_subset(conn, table, n, total):
    """Return *n* random IDs from DB (just the IDs, no data loaded)."""
    rng = np.random.RandomState(42)
    ids = rng.choice(total, size=n, replace=False) + 1  # DB ids are 1-based
    return ids.astype(np.int64)


def sample_batch_from_ids(conn, table, ids, batch_size, rng):
    """Sample *batch_size* random IDs from *ids*, query DB, return TrainInput."""
    idx = rng.choice(len(ids), size=batch_size, replace=True)
    batch_ids = sorted(int(ids[i]) for i in idx)
    placeholders = ",".join("?" for _ in batch_ids)
    rows = conn.execute(
        f"SELECT obs, mask, policy, value FROM {table} WHERE id IN ({placeholders})",
        batch_ids).fetchall()
    obs_l, mask_l, pol_l, val_l = [], [], [], []
    for obs_b, mask_b, pol_b, val_b in rows:
        obs_l.append(_unpack(obs_b, np.float32))
        mask_l.append(_unpack(mask_b, bool))
        pol_l.append(_unpack(pol_b, np.float32))
        val_l.append(_unpack(val_b, np.float32))
    return TrainInput(
        observation=np.stack(obs_l),
        legals_mask=np.stack(mask_l),
        policy=np.stack(pol_l),
        value=np.stack(val_l),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-lr", type=float, default=None,
                        help="Separate LR for policy head (None = use global LR)")
    parser.add_argument("--batches", type=int, default=NUM_BATCHES)
    parser.add_argument("--full", action="store_true",
                        help="Use full DB instead of subset")
    args = parser.parse_args()

    print(f"Loading model checkpoint-{CKPT_STEP} ...")
    model, cfg = load_model_and_config()
    print(f"  params={model.num_trainable_variables}")
    print(f"  config: lr={cfg['learning_rate']} wd={cfg['weight_decay']}"
          f"  batch={cfg['train_batch_size']}")

    sym = XiangqiSymmetry() if cfg.get("symmetry", 1) > 1 else None
    batch_size = cfg.get("train_batch_size", 256)
    entropy_weight = cfg.get("entropy_weight", 0.0)

    # Set up optimizer with optional per-layer LR
    if args.policy_lr is not None:
        policy_params = []
        other_params = []
        for name, p in model._model.named_parameters():
            if "policy_conv" in name:
                policy_params.append(p)
            else:
                other_params.append(p)
        model._optimizer = torch.optim.AdamW([
            {"params": other_params, "lr": cfg["learning_rate"],
             "weight_decay": cfg["weight_decay"]},
            {"params": policy_params, "lr": args.policy_lr,
             "weight_decay": cfg["weight_decay"]},
        ])
        print(f"  policy_lr={args.policy_lr}  (global lr={cfg['learning_rate']})")

    conn = sqlite3.connect(DB_PATH)
    table = "states"
    total = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"  DB rows: {total}")

    if args.full:
        subset_ids = None
        print(f"  Using full DB ({total} rows)")
    else:
        subset_n = min(SUBSET_SIZE, total)
        subset_ids = load_id_subset(conn, table, subset_n, total)
        print(f"  Subset: {len(subset_ids)} IDs (data stays on disk)")

    p_kl_hist = []
    policy_loss_hist = []
    v_kl_hist = []
    t0 = time.time()

    sample_rng = np.random.RandomState(123)
    for step in range(args.batches):
        if subset_ids is not None:
            batch = sample_batch_from_ids(conn, table, subset_ids,
                                          batch_size, sample_rng)
        else:
            ids = sorted(int(sample_rng.integers(1, total + 1, size=batch_size)))
            placeholders = ",".join("?" for _ in ids)
            rows = conn.execute(
                f"SELECT obs, mask, policy, value FROM states WHERE id IN ({placeholders})",
                ids).fetchall()
            obs_l, mask_l, pol_l, val_l = [], [], [], []
            for obs_b, mask_b, pol_b, val_b in rows:
                obs_l.append(_unpack(obs_b, np.float32))
                mask_l.append(_unpack(mask_b, bool))
                pol_l.append(_unpack(pol_b, np.float32))
                val_l.append(_unpack(val_b, np.float32))
            batch = TrainInput(
                observation=np.stack(obs_l), legals_mask=np.stack(mask_l),
                policy=np.stack(pol_l), value=np.stack(val_l))
        if sym is not None:
            obs, mask, policy, value = sym.augment_batch(
                batch.observation, batch.legals_mask,
                batch.policy, batch.value)
            batch = TrainInput(observation=obs, legals_mask=mask,
                               policy=policy, value=value)

        model._entropy_weight = entropy_weight
        loss = model.update(batch)
        p_kl_hist.append(loss.p_kl)
        policy_loss_hist.append(loss.policy)
        v_kl_hist.append(loss.v_kl)

        if (step + 1) % LOG_EVERY == 0:
            window_pkl = p_kl_hist[-LOG_EVERY:]
            window_vkl = v_kl_hist[-LOG_EVERY:]
            window_pl = policy_loss_hist[-LOG_EVERY:]
            elapsed = time.time() - t0
            print(f"  {step+1:4d}/{args.batches}  "
                  f"P-KL={np.mean(window_pkl):.4f}±{np.std(window_pkl):.4f}"
                  f"[{np.min(window_pkl):.4f},{np.max(window_pkl):.4f}]  "
                  f"V-KL={np.mean(window_vkl):.4f}±{np.std(window_vkl):.4f}"
                  f"[{np.min(window_vkl):.4f},{np.max(window_vkl):.4f}]  "
                  f"p_loss={np.mean(window_pl):.3f}  "
                  f"l2={loss.l2:.3f}  "
                  f"{elapsed:.0f}s")
    conn.close()
    elapsed = time.time() - t0

    print(f"\n{'='*50}")
    print(f"Done. {args.batches} batches in {elapsed:.0f}s"
          f" ({elapsed/args.batches:.1f}s/batch)")

    # Summary: head / tail P-KL and V-KL
    n = len(p_kl_hist)
    h = min(LOG_EVERY, n)
    p_head = np.mean(p_kl_hist[:h])
    p_tail = np.mean(p_kl_hist[-h:])
    p_change = (p_tail - p_head) / max(abs(p_head), 1e-9)
    v_head = np.mean(v_kl_hist[:h])
    v_tail = np.mean(v_kl_hist[-h:])
    v_change = (v_tail - v_head) / max(abs(v_head), 1e-9)
    print(f"  P-KL: {p_head:.4f} → {p_tail:.4f}  ({p_change:+.1%})")
    print(f"  V-KL: {v_head:.4f} → {v_tail:.4f}  ({v_change:+.1%})")
    if p_change < -0.1:
        print(f"  → P-KL decreasing — model IS learning")
    elif abs(p_change) < 0.05:
        print(f"  → P-KL flat — model STUCK on policy")
    else:
        print(f"  → P-KL increasing — model DIVERGING")

    # Save fitted model as checkpoint-<step>_fit.pt
    save_path = os.path.join(CKPT_DIR, f"checkpoint-{CKPT_STEP}_fit.pt")
    model._checkpoint_path = CKPT_DIR
    model.save_checkpoint(f"{CKPT_STEP}_fit")
    print(f"\n  Model saved to {save_path}")


if __name__ == "__main__":
    main()
