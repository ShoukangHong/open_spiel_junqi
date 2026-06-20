"""Shared data types for AlphaZero training — game-agnostic."""

from dataclasses import dataclass

import numpy as np


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
    grad_rel: float = 0.0
    update_rel: float = 0.0

    @property
    def total(self) -> float:
        return self.policy + self.value + self.l2

    def __str__(self) -> str:
        return (f"Losses(total: {self.total:.3f}, policy: {self.policy:.3f}, "
                f"value: {self.value:.3f}, l2: {self.l2:.3f}, "
                f"V-KL: {self.v_kl:.4f}, Top1: {self.top1:.1%}, "
                f"P-KL: {self.p_kl:.4f}, "
                f"GradRel: {self.grad_rel:.6f}, "
                f"UpdRel: {self.update_rel:.6f})")
