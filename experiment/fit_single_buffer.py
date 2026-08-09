"""Fit model to replay buffer DBs — disk-only multi-DB sampling.

Usage:
    python -m experiment.fit_single_buffer --config path/to/train_config.json
    python -m experiment.fit_single_buffer --config config.json --batches 5000
    python -m experiment.fit_single_buffer --config config.json --policy-lr 0.0001
"""

import argparse
import json
import os
import time

import numpy as np
import pyspiel
import torch

from train.core.model_builder import build_xiangqi_model
from train.core.replay_buffer import ReplayBuffer
from train.core.types import TrainInput
from train.model.xiangqi_symmetry import XiangqiSymmetry

LOG_EVERY = 100


def load_model_and_buffer(config_path, output_dir=None, fresh_optimizer=False):
    """Load config, game, model from latest checkpoint (or fresh), and replay buffer."""
    with open(config_path) as f:
        cfg = json.load(f)

    game = pyspiel.load_game("xiangqi")
    model = build_xiangqi_model(game, {
        "nn_width": cfg["nn_width"],
        "nn_depth": cfg["nn_depth"],
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "path": output_dir or cfg["path"],
    })

    # Resume from latest checkpoint
    ckpt_dir = output_dir or cfg["path"]
    start_step = _find_latest(ckpt_dir)
    if start_step > 0 and not fresh_optimizer:
        model.load_checkpoint(start_step)
        print(f"  Resumed from checkpoint-{start_step}")
    elif start_step > 0:
        model.load_checkpoint(start_step, fresh_optimizer=True)
        print(f"  Resumed from checkpoint-{start_step} (fresh Adam)")
    else:
        print("  Starting fresh (no checkpoint found)")

    # Open replay buffer — disk-only, all DBs under cfg.path
    buf = ReplayBuffer(
        max_size=10**12,  # unbounded: sample from all rows
        db_path=os.path.join(cfg["path"], "buffer.db"),
        max_db_rows=10**12,
    )
    n_dbs = len(buf._all_db_paths())
    total_mb = sum(os.path.getsize(p) for p in buf._all_db_paths()
                   if os.path.exists(p)) / 1e6
    print(f"  Buffer: {len(buf)} states ({n_dbs} DB files, {total_mb:.0f} MB)")
    return model, cfg, buf, start_step


def _find_latest(directory):
    """Return the largest checkpoint step number in *directory*, or 0."""
    best = 0
    if not os.path.isdir(directory):
        return 0
    for name in os.listdir(directory):
        if name.startswith("checkpoint-") and name.endswith(".pt"):
            try:
                step = int(name[len("checkpoint-"):-len(".pt")])
                if step > best:
                    best = step
            except ValueError:
                pass
    return best


def main():
    parser = argparse.ArgumentParser(description="Fit model to buffer DBs (disk-only)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to train_config.json")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for checkpoints (default: config.path)")
    parser.add_argument("--batches", type=int, default=5000,
                        help="Total forward passes")
    parser.add_argument("--checkpoint-every", type=int, default=1000,
                        help="Save checkpoint every N forward passes")
    parser.add_argument("--log-every", type=int, default=100,
                        help="Log stats every N forward passes")
    parser.add_argument("--policy-lr", type=float, default=None,
                        help="Separate LR for policy head")
    parser.add_argument("--value-learn-prob", type=float, default=None,
                        help="Probability of training value head each batch")
    parser.add_argument("--reset-optimizer", action="store_true",
                        help="Start with fresh Adam state")
    args = parser.parse_args()

    # ── Load ────────────────────────────────────────────────────────────
    print(f"Loading config from {args.config} ...")
    model, cfg, buf, start_step = load_model_and_buffer(
        args.config, args.output,
        fresh_optimizer=args.reset_optimizer)

    print(f"  params={model.num_trainable_variables}")
    print(f"  config: lr={cfg['learning_rate']} wd={cfg['weight_decay']}"
          f"  batch={cfg.get('train_batch_size', 256)}")
    print(f"  batches={args.batches}  ckpt_every={args.checkpoint_every}")

    sym = XiangqiSymmetry() if cfg.get("symmetry", 1) > 1 else None
    batch_size = cfg.get("train_batch_size", 256)
    entropy_weight = cfg.get("entropy_weight", 0.0)
    value_learn_prob = args.value_learn_prob
    if value_learn_prob is None:
        value_learn_prob = cfg.get("value_learn_prob", 1.0)

    # ── Optional per-layer LR ───────────────────────────────────────────
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

    # ── Training loop ───────────────────────────────────────────────────
    p_kl_hist = []; v_kl_hist = []; l2_hist = []
    total_updates = start_step
    last_ckpt = start_step

    ckpt_dir = args.output or cfg["path"]
    model._checkpoint_path = ckpt_dir
    if start_step == 0:
        model.save_checkpoint(save_optimizer=False, step=0)
        print(f"\n  Initial checkpoint saved: {os.path.join(ckpt_dir, 'checkpoint-0.pt')}")
    print(f"Training {args.batches} forward passes (from step {start_step}) ...")
    t0 = time.time()

    for _ in range(args.batches):
        batch = buf.sample_uniform(batch_size)
        if sym is not None:
            obs, mask, policy, value = sym.augment_batch(
                batch.observation, batch.legals_mask,
                batch.policy, batch.value)
            batch = TrainInput(observation=obs, legals_mask=mask,
                               policy=policy, value=value)

        model._entropy_weight = entropy_weight
        model._value_learn_prob = value_learn_prob
        loss = model.update(batch)
        p_kl_hist.append(loss.p_kl)
        v_kl_hist.append(loss.v_kl)
        l2_hist.append(loss.l2)
        total_updates += 1

        # Checkpoint
        if total_updates - last_ckpt >= args.checkpoint_every:
            ckpt_dir = args.output or cfg["path"]
            model._checkpoint_path = ckpt_dir
            model.save_checkpoint(save_optimizer=False, step=total_updates)
            print(f"  [checkpoint] forward={total_updates}")
            last_ckpt = total_updates

        # Logging
        if (total_updates) % args.log_every == 0:
            w = args.log_every
            window_pkl = p_kl_hist[-w:]
            window_vkl = v_kl_hist[-w:]
            window_l2 = l2_hist[-w:]
            elapsed = time.time() - t0
            print(f"  {total_updates:5d}/{args.batches}  "
                  f"P-KL={np.mean(window_pkl):.4f}±{np.std(window_pkl):.4f}"
                  f"[{np.min(window_pkl):.4f},{np.max(window_pkl):.4f}]  "
                  f"V-KL={np.mean(window_vkl):.4f}±{np.std(window_vkl):.4f}"
                  f"[{np.min(window_vkl):.4f},{np.max(window_vkl):.4f}]  "
                  f"L2={np.mean(window_l2):.3f}  "
                  f"{elapsed:.0f}s")

    buf.close()
    elapsed = time.time() - t0

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Done. {args.batches} batches in {elapsed:.0f}s"
          f" ({elapsed/args.batches*1000:.1f}ms/batch)")

    n = len(p_kl_hist)
    h = min(args.log_every, n)
    p_head = np.mean(p_kl_hist[:h])
    p_tail = np.mean(p_kl_hist[-h:])
    p_change = (p_tail - p_head) / max(abs(p_head), 1e-9)
    v_head = np.mean(v_kl_hist[:h])
    v_tail = np.mean(v_kl_hist[-h:])
    v_change = (v_tail - v_head) / max(abs(v_head), 1e-9)
    l2_head = np.mean(l2_hist[:h])
    l2_tail = np.mean(l2_hist[-h:])
    print(f"  P-KL: {p_head:.4f} → {p_tail:.4f}  ({p_change:+.1%})")
    print(f"  V-KL: {v_head:.4f} → {v_tail:.4f}  ({v_change:+.1%})")
    print(f"  L2:   {l2_head:.4f} → {l2_tail:.4f}")

    if p_change < -0.1:
        print(f"  → P-KL decreasing — model IS learning")
    elif abs(p_change) < 0.05:
        print(f"  → P-KL flat — model STUCK on policy")
    else:
        print(f"  → P-KL increasing — model DIVERGING")

    # ── Save fitted model ───────────────────────────────────────────────
    ckpt_dir = args.output or cfg["path"]
    model._checkpoint_path = ckpt_dir
    model.save_checkpoint(save_optimizer=False, step=f"fit_{total_updates}")
    print(f"  Model saved to {os.path.join(ckpt_dir, f'checkpoint-fit_{total_updates}.pt')}")


if __name__ == "__main__":
    main()
