"""PyTorch AlphaZero-style ResNet for Othello.

Input:  observation tensor  (batch, 4, 8, 8)
Output: policy logits      (batch, 65) — 64 squares + pass
        value               (batch,)   — tanh → [-1, 1]
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train.model.blocks import ConvBlock, ResBlock
from train.model.model import Model  # noqa: F401 — re-exported for backward compat


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
                if name == "policy_fc":
                    nn.init.uniform_(m.weight, -0.03, 0.03)
                elif name == "value_fc2":
                    nn.init.zeros_(m.weight)
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

    def batch_forward_raw(self, observations: np.ndarray) -> np.ndarray:
        """Forward only — returns (policy_logits, value_logits), no softmax."""
        self.eval()
        with torch.no_grad():
            obs_t = torch.from_numpy(
                np.ascontiguousarray(observations, dtype=np.float32)).to(self.device)
            if not torch.isfinite(obs_t).all():
                raise RuntimeError("batch_forward_raw: obs_t contains NaN/Inf")
            if obs_t.dim() == 2:
                obs_t = obs_t.reshape(obs_t.shape[0], self.input_channels,
                                      self.board_size, self.board_size)
            policy_logits, value = self.forward(obs_t)
            if not torch.isfinite(policy_logits).all():
                raise RuntimeError("batch_forward_raw: NaN in policy_logits")
            if not torch.isfinite(value).all():
                raise RuntimeError("batch_forward_raw: NaN in value")
            return policy_logits.cpu().numpy(), value.cpu().numpy()

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


