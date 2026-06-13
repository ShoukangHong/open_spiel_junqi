"""PyTorch AlphaZero-style ResNet for Xiangqi (Chinese Chess).

Input:  observation tensor  (batch, 17, 10, 9)
Output: policy               (batch, 8100) — plane encoding: 90 src planes × 90 tgt cells
        value                (batch, 3)    — WDL logits
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train.model.blocks import ConvBlock, ResBlock
from train.model.model import Model  # noqa: F401 — re-exported for convenience


class XiangqiResNet(nn.Module):
    """AlphaZero ResNet for Xiangqi with plane-encoding policy head.

    Policy head outputs 90 planes (one per source square), each 10×9 spatial
    (target position).  Reshaped to flat 8100-way for masked softmax.
    """

    def __init__(self, input_channels: int = 17, board_rows: int = 10,
                 board_cols: int = 9, output_size: int = 8100,
                 nn_width: int = 32, nn_depth: int = 5):
        super().__init__()
        self.input_channels = input_channels
        self.board_rows = board_rows
        self.board_cols = board_cols
        self.output_size = output_size
        self.nn_width = nn_width
        self.nn_depth = nn_depth
        num_src = board_rows * board_cols  # 90

        # Torso
        self.conv_in = ConvBlock(input_channels, nn_width, kernel_size=3)
        self.res_blocks = nn.Sequential(
            *[ResBlock(nn_width) for _ in range(nn_depth)]
        )

        # Policy head — plane encoding: 1 plane per source square
        self.policy_conv = nn.Conv2d(nn_width, num_src, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(num_src)

        # Value head — WDL 3-class
        self.value_conv = nn.Conv2d(nn_width, 4, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(4)
        self.value_fc1_in = 4 * board_rows * board_cols  # 360
        self.value_fc1 = nn.Linear(self.value_fc1_in, nn_width)
        self.value_fc2 = nn.Linear(nn_width, 3)

        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if name in ("policy_conv", "value_fc2"):
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
        batch = x.shape[0]

        # Torso
        x = self.conv_in(x)
        x = self.res_blocks(x)

        # Policy head: 90 planes × (10, 9) → flat 8100
        p = F.relu(self.policy_bn(self.policy_conv(x)))   # (batch, 90, 10, 9)
        policy_logits = p.reshape(batch, -1)                # (batch, 8100)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))       # (batch, 4, 10, 9)
        v = v.reshape(batch, -1)                             # (batch, 360)
        v = F.relu(self.value_fc1(v))
        value = self.value_fc2(v)                            # (batch, 3)

        return policy_logits, value

    def inference(self, observation: np.ndarray,
                  legals_mask: np.ndarray) -> tuple:
        """Single-sample inference (for MCTS).

        Returns:
            value: (3,) numpy array (softmaxed WDL).
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
                                      self.board_rows, self.board_cols)

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
        """Forward only — returns (policy_logits, value_logits), no softmax.

        For server-side inference where softmax is deferred to scatter phase.
        """
        self.eval()
        with torch.no_grad():
            obs_t = torch.from_numpy(
                np.ascontiguousarray(observations, dtype=np.float32)).to(self.device)
            if not torch.isfinite(obs_t).all():
                raise RuntimeError("batch_forward_raw: obs_t contains NaN/Inf")
            if obs_t.dim() == 2:
                obs_t = obs_t.reshape(obs_t.shape[0], self.input_channels,
                                      self.board_rows, self.board_cols)
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
            observations: (batch, ...) — flat (batch, 1350) or shaped
                (batch, 15, 10, 9).
            legals_masks: (batch, 8100).

        Returns:
            values: (batch, 3) numpy array
            policies: (batch, 8100) numpy array
        """
        self.eval()
        with torch.no_grad():
            obs_t = torch.from_numpy(
                np.ascontiguousarray(observations, dtype=np.float32)).to(self.device)
            mask_t = torch.from_numpy(
                np.asarray(legals_masks, dtype=bool)).to(self.device)

            if not torch.isfinite(obs_t).all():
                raise RuntimeError("batch_inference: obs_t contains NaN/Inf")

            if obs_t.dim() == 2:
                obs_t = obs_t.reshape(obs_t.shape[0], self.input_channels,
                                      self.board_rows, self.board_cols)

            policy_logits, value = self.forward(obs_t)

            if not torch.isfinite(policy_logits).all():
                raise RuntimeError("batch_inference: policy_logits NaN/Inf")
            if not torch.isfinite(value).all():
                raise RuntimeError("batch_inference: value NaN/Inf")

            policy_logits = torch.clamp(policy_logits, -30, 30)
            policy_logits = torch.where(mask_t, policy_logits,
                                        torch.full_like(policy_logits, -1e9))
            policies = F.softmax(policy_logits, dim=-1)
            policies = policies * mask_t
            policies = policies / policies.sum(dim=-1, keepdims=True).clamp(min=1e-9)

            value = F.softmax(value, dim=-1)
            return value.cpu().numpy(), policies.cpu().numpy()
