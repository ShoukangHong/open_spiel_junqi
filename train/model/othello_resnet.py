"""PyTorch AlphaZero-style ResNet for Othello.

Input:  observation tensor  (batch, 4, 8, 8)
Output: policy logits      (batch, 65) — 64 squares + pass
        value               (batch,)   — tanh → [-1, 1]
"""

from dataclasses import dataclass
import os
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Data structures (mirror open_spiel alpha_zero utils) ────────────────────

@dataclass
class TrainInput:
    observation: np.ndarray
    legals_mask: np.ndarray
    policy: np.ndarray
    value: np.ndarray


@dataclass
class Losses:
    policy: float
    value: float
    l2: float
    v_kl: float = 0.0
    top1: float = 0.0
    p_kl: float = 0.0

    @property
    def total(self) -> float:
        return self.policy + self.value + self.l2

    def __str__(self) -> str:
        return (f"Losses(total: {self.total:.3f}, policy: {self.policy:.3f}, "
                f"value: {self.value:.3f}, l2: {self.l2:.3f}, "
                f"V-KL: {self.v_kl:.4f}, Top1: {self.top1:.1%}, "
                f"P-KL: {self.p_kl:.4f})")


# ── Building blocks ─────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv2d + BatchNorm + ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    """Residual block: Conv→BN→ReLU→Conv→BN + skip → ReLU."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


# ── Main model ──────────────────────────────────────────────────────────────

class OthelloResNet(nn.Module):
    """AlphaZero ResNet for Othello.

    Args:
        input_channels: observation tensor channels (4 for Othello: empty, black, white, player_to_move).
        board_size: spatial size (8 for Othello).
        output_size: number of distinct actions (65 for Othello).
        nn_width: number of filters in conv layers.
        nn_depth: number of residual blocks.
    """

    def __init__(self, input_channels: int = 4, board_size: int = 8,
                 output_size: int = 65, nn_width: int = 32, nn_depth: int = 5):
        super().__init__()
        self.input_channels = input_channels
        self.board_size = board_size
        self.output_size = output_size
        self.nn_width = nn_width
        self.nn_depth = nn_depth

        # Torso
        self.conv_in = ConvBlock(input_channels, nn_width, kernel_size=3)
        self.res_blocks = nn.Sequential(
            *[ResBlock(nn_width) for _ in range(nn_depth)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(nn_width, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, output_size)

        # Value head — WDL: 3 output classes (win, draw, loss)
        self.value_conv = nn.Conv2d(nn_width, 4, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(4)
        self.value_fc1 = nn.Linear(4 * board_size * board_size, nn_width)
        self.value_fc2 = nn.Linear(nn_width, 3)

        self._init_weights()

    def _init_weights(self):
        """Explicit kaiming init for all conv/linear layers."""
        for name, m in self.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if name in ("policy_fc", "value_fc2"):
                    nn.init.uniform_(m.weight, -0.03, 0.03)
                else:
                    nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                            nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        """Forward pass.

        Returns:
            policy_logits: (batch, output_size)
            value: (batch, 3) — WDL logits (no softmax).
        """
        batch = x.shape[0]

        # Torso
        x = self.conv_in(x)
        x = self.res_blocks(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.reshape(batch, -1)
        policy_logits = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.reshape(batch, -1)
        v = F.relu(self.value_fc1(v))
        value = self.value_fc2(v)  # (batch, 3) logits

        return policy_logits, value

    def inference(self, observation: np.ndarray,
                  legals_mask: np.ndarray) -> tuple:
        """Single-sample inference (for MCTS).

        Args:
            observation: (input_channels, board_size, board_size) or flat.
            legals_mask: (output_size,) bool array.

        Returns:
            value: float
            policy: (output_size,) numpy array (softmax over legal actions).
        """
        self.eval()
        with torch.no_grad():
            obs_t = torch.from_numpy(
                np.ascontiguousarray(observation, dtype=np.float32)).to(self.device)
            if obs_t.dim() == 3:
                obs_t = obs_t.unsqueeze(0)
            else:
                obs_t = obs_t.reshape(1, self.input_channels,
                                      self.board_size, self.board_size)

            mask_t = torch.from_numpy(
                np.asarray(legals_mask, dtype=bool)).to(self.device)
            if mask_t.dim() == 1:
                mask_t = mask_t.unsqueeze(0)

            policy_logits, value = self.forward(obs_t)
            policy_logits = torch.clamp(policy_logits, -30, 30)
            policy_logits = torch.where(mask_t, policy_logits,
                                        torch.full_like(policy_logits, -1e9))
            policy = F.softmax(policy_logits, dim=-1)
            policy = policy * mask_t
            policy = policy / policy.sum(dim=-1, keepdims=True).clamp(min=1e-9)

            val = F.softmax(value, dim=-1)[0].cpu().numpy()
            return val, policy[0].cpu().numpy()

    def batch_inference(self, observations: np.ndarray,
                        legals_masks: np.ndarray) -> tuple:
        """Batch inference.

        Args:
            observations: (batch, ...) — flat (batch, 256) or shaped
                (batch, input_channels, board_size, board_size).
            legals_masks: (batch, output_size).

        Returns:
            values: (batch,) numpy array
            policies: (batch, output_size) numpy array
        """
        self.eval()
        with torch.no_grad():
            obs_t = torch.from_numpy(
                np.ascontiguousarray(observations, dtype=np.float32)).to(self.device)
            mask_t = torch.from_numpy(
                np.asarray(legals_masks, dtype=bool)).to(self.device)

            # NaN/Inf guard: corrupted weights produce NaN logits,
            # which cause CUDA unknown error in downstream ops.
            if not torch.isfinite(obs_t).all():
                raise RuntimeError("batch_inference: obs_t contains NaN/Inf")

            # pyspiel returns flat observations — reshape to (C, H, W)
            if obs_t.dim() == 2:
                obs_t = obs_t.reshape(obs_t.shape[0], self.input_channels,
                                      self.board_size, self.board_size)

            policy_logits, value = self.forward(obs_t)

            if not torch.isfinite(policy_logits).all():
                raise RuntimeError("batch_inference: policy_logits contains NaN/Inf (model weights likely corrupted)")
            if not torch.isfinite(value).all():
                raise RuntimeError("batch_inference: value contains NaN/Inf (model weights likely corrupted)")

            policy_logits = torch.clamp(policy_logits, -30, 30)
            policy_logits = torch.where(mask_t, policy_logits,
                                        torch.full_like(policy_logits, -1e9))
            policies = F.softmax(policy_logits, dim=-1)
            policies = policies * mask_t
            policies = policies / policies.sum(dim=-1, keepdims=True).clamp(min=1e-9)

            value = F.softmax(value, dim=-1)
            return value.cpu().numpy(), policies.cpu().numpy()


# ── Training wrapper ────────────────────────────────────────────────────────

class Model:
    """Training wrapper around OthelloResNet."""

    def __init__(self, model: OthelloResNet, learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4, device: str = "cpu",
                 checkpoint_path: Optional[str] = None):
        if device == "cuda" and not torch.cuda.is_available():
            print("[model] CUDA not available, falling back to CPU")
            device = "cpu"
        self._model = model.to(device)
        self._device = device
        self._weight_decay = weight_decay
        self._checkpoint_path = checkpoint_path

        # Separate weight decay for non-bias/non-bn params
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if "bn" in name or "bias" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        self._lr = learning_rate
        self._optimizer = torch.optim.AdamW([
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=learning_rate)

    @property
    def num_trainable_variables(self) -> int:
        return sum(p.numel() for p in self._model.parameters())

    def train(self):
        self._model.train()

    def eval(self):
        self._model.eval()

    def inference(self, observation: np.ndarray,
                  legals_mask: np.ndarray) -> tuple:
        return self._model.inference(observation, legals_mask)

    def batch_inference(self, observations: np.ndarray,
                        legals_masks: np.ndarray) -> tuple:
        return self._model.batch_inference(observations, legals_masks)

    def update(self, batch: TrainInput) -> Losses:
        """Single gradient update step.

        Args:
            batch: TrainInput with stacked observations, masks, policies, values.

        Returns:
            Losses with float values.
        """
        self._model.train()

        obs = torch.from_numpy(
            np.ascontiguousarray(batch.observation, dtype=np.float32))
        mask = torch.from_numpy(np.asarray(batch.legals_mask, dtype=bool))
        target_policy = torch.from_numpy(
            np.ascontiguousarray(batch.policy, dtype=np.float32))
        target_value = torch.from_numpy(np.asarray(batch.value, dtype=np.float32))

        # Ensure policy targets are normalized
        target_policy = target_policy / target_policy.sum(dim=-1, keepdims=True).clamp(min=1e-9)

        dev = self._device if self._device != "cpu" else self._model.device
        obs = obs.to(dev)
        mask = mask.to(dev)
        target_policy = target_policy.to(dev)
        target_value = target_value.to(dev)

        # Reshape flat observations from buffer to (batch, C, H, W)
        if obs.dim() == 2:
            obs = obs.reshape(obs.shape[0], self._model.input_channels,
                              self._model.board_size, self._model.board_size)

        policy_logits, value_pred = self._model(obs)

        # AlphaZero policy loss: full soft cross-entropy with MCTS visit
        # distribution (equivalent to optax.softmax_cross_entropy).
        policy_logits = torch.clamp(policy_logits, -30, 30)
        policy_logits_masked = torch.where(
            mask, policy_logits,
            torch.full_like(policy_logits, -1e9))
        log_probs = F.log_softmax(policy_logits_masked, dim=-1)
        policy_loss = -(target_policy * log_probs).sum(dim=-1).mean()

        # Policy metrics (interpretable)
        with torch.no_grad():
            policy_pred = F.softmax(policy_logits_masked, dim=-1)
            top1 = (policy_pred.argmax(-1) == target_policy.argmax(-1)
                    ).float().mean().item()
            eps = 1e-12
            p_kl = (target_policy * (torch.log(target_policy + eps)
                                     - torch.log(policy_pred + eps))
                    ).sum(dim=-1).mean().item()

        # Value loss: cross-entropy with soft WDL target
        value_loss = -(target_value *
                       F.log_softmax(value_pred, dim=-1)
                       ).sum(dim=-1).mean()

        # WDL KL divergence
        with torch.no_grad():
            wdl_pred = F.softmax(value_pred, dim=-1)
            v_kl = (target_value * (torch.log(target_value + eps)
                                    - torch.log(wdl_pred + eps))
                    ).sum(dim=-1).mean().item()

        # Track L2 loss contribution (matching JAX formula)
        l2_reg = sum(
            (p ** 2).sum()
            for name, p in self._model.named_parameters()
            if "bn" not in name and "bias" not in name
        ).item() * self._weight_decay

        total_loss = policy_loss + value_loss

        self._optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
        self._optimizer.step()

        return Losses(policy=policy_loss.item(), value=value_loss.item(),
                      l2=l2_reg, v_kl=v_kl, top1=top1, p_kl=p_kl)

    def save_checkpoint(self, step: int) -> str:
        if not self._checkpoint_path:
            return ""
        os.makedirs(self._checkpoint_path, exist_ok=True)
        # Check for NaN/Inf before saving — prevents persisting corrupted weights.
        for name, p in self._model.named_parameters():
            if not torch.isfinite(p).all():
                raise RuntimeError(
                    f"save_checkpoint: parameter '{name}' contains NaN/Inf "
                    f"at step {step} — training diverged")
        filepath = os.path.join(self._checkpoint_path, f"checkpoint-{step}.pt")
        # Write to temp then rename atomically so concurrent readers never
        # see a half-written file (which can corrupt CUDA context on load).
        tmp = filepath + ".tmp"
        torch.save({
            "step": step,
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
        }, tmp)
        os.replace(tmp, filepath)  # atomic on Windows
        return filepath

    def load_checkpoint(self, step: int) -> None:
        if not self._checkpoint_path:
            return
        filepath = os.path.join(self._checkpoint_path, f"checkpoint-{step}.pt")
        if os.path.exists(filepath):
            ckpt = torch.load(filepath, map_location=self._device, weights_only=False)
            self._model.load_state_dict(ckpt["model_state_dict"],
                                        strict=False)
            self._optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            # Keep LR from the current config, not from the checkpoint
            for pg in self._optimizer.param_groups:
                pg["lr"] = self._lr
        else:
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
