"""Shared training wrapper around PyTorch nn.Module — game-agnostic.

Wraps any ResNet-style model that provides:
  - self.input_channels, self.board_rows, self.board_cols (or self.board_size)
  - inference(obs, mask) → (value, policy)
  - batch_inference(obs, mask) → (values, policies)
  - forward(x) → (policy_logits, value)
"""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from train.core.types import TrainInput, Losses


class Model:
    """Training wrapper around a PyTorch nn.Module."""

    def __init__(self, model: torch.nn.Module, learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4, device: str = "cpu",
                 checkpoint_path: Optional[str] = None):
        if device == "cuda" and not torch.cuda.is_available():
            print("[model] CUDA not available, falling back to CPU")
            device = "cpu"
        self._model = model.to(device)
        self._device = device
        self._weight_decay = weight_decay
        self._checkpoint_path = checkpoint_path

        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if "bn" in name or "bias" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        self._lr = learning_rate
        self._entropy_weight = 0.0
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

    def batch_forward_raw(self, observations: np.ndarray) -> np.ndarray:
        return self._model.batch_forward_raw(observations)

    def update(self, batch: TrainInput) -> Losses:
        self._model.train()

        obs = torch.from_numpy(
            np.ascontiguousarray(batch.observation, dtype=np.float32))
        mask = torch.from_numpy(np.asarray(batch.legals_mask, dtype=bool))
        target_policy = torch.from_numpy(
            np.ascontiguousarray(batch.policy, dtype=np.float32))
        target_value = torch.from_numpy(np.asarray(batch.value, dtype=np.float32))

        target_policy = target_policy / target_policy.sum(dim=-1, keepdims=True).clamp(min=1e-9)

        dev = self._device if self._device != "cpu" else self._model.device
        obs = obs.to(dev)
        mask = mask.to(dev)
        target_policy = target_policy.to(dev) * mask  # zero out illegal actions
        target_value = target_value.to(dev)

        # Reshape flat observations to (batch, C, H, W) using model's spatial dims
        if obs.dim() == 2:
            h = getattr(self._model, 'board_rows',
                        getattr(self._model, 'board_size', None))
            w = getattr(self._model, 'board_cols',
                        getattr(self._model, 'board_size', None))
            obs = obs.reshape(obs.shape[0], self._model.input_channels, h, w)

        policy_logits, value_pred = self._model(obs)

        # Policy loss: cross-entropy with MCTS visit distribution
        policy_logits = torch.clamp(policy_logits, -30, 30)
        policy_logits_masked = torch.where(
            mask, policy_logits,
            torch.full_like(policy_logits, -1e9))
        log_probs = F.log_softmax(policy_logits_masked, dim=-1)

        with torch.no_grad():
            p_weights = torch.ones(target_value.shape[0], device=dev)
            p_weights[target_value[:, 2] > 0.99999] = 0.2
        per_sample = -(target_policy * log_probs).sum(dim=-1)
        policy_loss = (p_weights * per_sample).sum() / p_weights.sum().clamp(min=1)

        eps = 1e-12
        with torch.no_grad():
            policy_pred = F.softmax(policy_logits_masked, dim=-1)
            top1 = (policy_pred.argmax(-1) == target_policy.argmax(-1)
                    ).float().mean().item()
            p_ce = -(target_policy * log_probs).sum(dim=-1)
            p_ent = -(target_policy * torch.log(target_policy + eps)).sum(dim=-1)
            p_kl = ((p_weights * (p_ce - p_ent)).sum()
                    / p_weights.sum().clamp(min=1)).item()

        # Value loss: cross-entropy with soft WDL target
        log_val = F.log_softmax(value_pred, dim=-1)
        value_loss = -(target_value * log_val).sum(dim=-1).mean()

        with torch.no_grad():
            v_ce = -(target_value * log_val).sum(dim=-1)
            v_ent = -(target_value * torch.log(target_value + eps)).sum(dim=-1)
            v_kl = (v_ce - v_ent).mean().item()

        l2_reg = sum(
            (p ** 2).sum()
            for name, p in self._model.named_parameters()
            if "bn" not in name and "bias" not in name
        ).item() * self._weight_decay

        ent_bonus = -(policy_pred * log_probs).sum(dim=-1).mean()

        total_loss = policy_loss + value_loss
        if self._entropy_weight > 0:
            total_loss = total_loss - self._entropy_weight * ent_bonus

        self._optimizer.zero_grad()
        total_loss.backward()

        # ── Relative gradient / update norms (pre-clip) ───────────────────
        with torch.no_grad():
            w_sq = 0.0
            g_sq = 0.0
            for p in self._model.parameters():
                w_sq += (p.detach() ** 2).sum().item()
                if p.grad is not None:
                    g_sq += (p.grad.detach() ** 2).sum().item()
            w_norm = w_sq ** 0.5
            g_norm = g_sq ** 0.5
            grad_rel = g_norm / max(w_norm, 1e-12)
            w_old = torch.cat([p.detach().flatten()
                               for p in self._model.parameters()])

        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
        self._optimizer.step()

        with torch.no_grad():
            w_new = torch.cat([p.detach().flatten()
                               for p in self._model.parameters()])
            delta_norm = (w_new - w_old).norm().item()
            update_rel = delta_norm / max(w_norm, 1e-12)

        # Catch corrupted BN running stats immediately (not just at checkpoint)
        for name, b in self._model.named_buffers():
            if not torch.isfinite(b).all():
                # Dump the offending batch for offline investigation
                import time as _time
                dump = f"nan_batch_{_time.strftime('%Y%m%d_%H%M%S')}.npz"
                obs_np = (batch.observation[:1].cpu().numpy()
                          if isinstance(batch.observation, torch.Tensor)
                          else batch.observation[:1])
                np.savez_compressed(dump, obs=obs_np,
                                    mask=batch.legals_mask[:1],
                                    policy=batch.policy[:1],
                                    value=batch.value[:1])
                raise RuntimeError(
                    f"training step: buffer '{name}' became NaN/Inf — "
                    f"dumped first sample to {dump}")

        return Losses(policy=policy_loss.item(), value=value_loss.item(),
                      l2=l2_reg, v_kl=v_kl, top1=top1, p_kl=p_kl,
                      grad_rel=grad_rel, update_rel=update_rel)

    def save_checkpoint(self, step: int) -> str:
        if not self._checkpoint_path:
            return ""
        os.makedirs(self._checkpoint_path, exist_ok=True)
        for name, p in self._model.named_parameters():
            if not torch.isfinite(p).all():
                raise RuntimeError(
                    f"save_checkpoint: parameter '{name}' contains NaN/Inf "
                    f"at step {step} — training diverged")
        for name, b in self._model.named_buffers():
            if not torch.isfinite(b).all():
                raise RuntimeError(
                    f"save_checkpoint: buffer '{name}' contains NaN/Inf "
                    f"at step {step} — training diverged")
        filepath = os.path.join(self._checkpoint_path, f"checkpoint-{step}.pt")
        tmp = filepath + ".tmp"
        torch.save({
            "step": step,
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
        }, tmp)
        os.replace(tmp, filepath)
        return filepath

    def load_checkpoint(self, step: int) -> None:
        if not self._checkpoint_path:
            return
        filepath = os.path.join(self._checkpoint_path, f"checkpoint-{step}.pt")
        if os.path.exists(filepath):
            ckpt = torch.load(filepath, map_location=self._device, weights_only=False)
            self._model.load_state_dict(ckpt["model_state_dict"], strict=False)
            self._optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            for pg in self._optimizer.param_groups:
                pg["lr"] = self._lr
        else:
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
