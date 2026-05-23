"""Quick buffer-fit check: can the model learn the data it was trained on?"""

import json
import os
import sys

import numpy as np
import torch
import pyspiel

from train.model.othello_resnet import Model, OthelloResNet, TrainInput, Losses

CHECKPOINT_DIR = r"C:\Users\shouk\othello_train_v2"
CHECKPOINT_STEP = 5
BUFFER_FILE = f"buffer-checkpoint-{CHECKPOINT_STEP}.npz"
BATCH_SIZE = 128
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4


def main():
    # ── Load config ───────────────────────────────────────────────────────
    with open(os.path.join(CHECKPOINT_DIR, "train_config.json")) as f:
        train_cfg = json.load(f)
    nn_width = train_cfg["nn_width"]
    nn_depth = train_cfg["nn_depth"]
    device = train_cfg.get("device", "cpu")

    # ── Load buffer ───────────────────────────────────────────────────────
    buf_path = os.path.join(CHECKPOINT_DIR, BUFFER_FILE)
    data = np.load(buf_path)
    obs = data["obs"]
    masks = data["masks"]
    policies = data["policies"]
    values = data["values"]
    N_raw = len(values)
    print(f"Buffer: {N_raw} raw samples  obs_shape={obs.shape}")

    # Deduplicate: same obs+mask → keep first occurrence.
    # Different training iterations may assign different targets to the same
    # state, making perfect fit impossible.  Dedup gives a clean target set.
    seen = {}
    keep_idx = []
    for i in range(N_raw):
        key = obs[i].tobytes() + masks[i].tobytes()
        if key not in seen:
            seen[key] = i
            keep_idx.append(i)
    obs = obs[keep_idx]
    masks = masks[keep_idx]
    policies = policies[keep_idx]
    values = values[keep_idx]
    N = len(values)
    print(f"Deduplicated: {N} unique states  (removed {N_raw - N} duplicates)")

    # ── Build model ───────────────────────────────────────────────────────
    game = pyspiel.load_game("othello")
    obs_shape = game.observation_tensor_shape()
    num_actions = game.num_distinct_actions()

    net = OthelloResNet(
        input_channels=obs_shape[0], board_size=obs_shape[1],
        output_size=num_actions, nn_width=nn_width, nn_depth=nn_depth)
    model = Model(net, learning_rate=LR, weight_decay=WEIGHT_DECAY,
                  device=device, checkpoint_path=CHECKPOINT_DIR)

    # Start from checkpoint
    model.load_checkpoint(CHECKPOINT_STEP)
    print(f"Loaded checkpoint-{CHECKPOINT_STEP}  params={model.num_trainable_variables}")

    # Train on buffer
    steps_per_epoch = max(1, N // BATCH_SIZE)
    print(f"\nTraining: {EPOCHS} epochs × {steps_per_epoch} steps/epoch"
          f"  batch={BATCH_SIZE}  device={device}\n")
    print(f"{'epoch':>5s}  {'policy_loss':>11s}  {'value_loss':>10s}  {'total':>8s}")
    print("-" * 42)

    for epoch in range(1, EPOCHS + 1):
        epoch_losses = []
        # Shuffle indices each epoch
        idx = np.random.permutation(N)
        for start in range(0, N, BATCH_SIZE):
            batch_idx = idx[start:start + BATCH_SIZE]
            batch = TrainInput(
                observation=obs[batch_idx],
                legals_mask=masks[batch_idx],
                policy=policies[batch_idx],
                value=values[batch_idx],
            )
            loss = model.update(batch)
            epoch_losses.append(loss)

        avg = Losses(
            policy=sum(l.policy for l in epoch_losses) / len(epoch_losses),
            value=sum(l.value for l in epoch_losses) / len(epoch_losses),
            l2=sum(l.l2 for l in epoch_losses) / len(epoch_losses),
        )
        print(f"{epoch:5d}  {avg.policy:11.4f}  {avg.value:10.4f}  {avg.total:8.4f}")

    print(f"\nDone. Final loss={avg}")


if __name__ == "__main__":
    main()
