"""Visualize the weak-move Boltzmann weight distribution.

Usage:
    python experiment/boltzmann_weights.py
"""

import numpy as np
import matplotlib.pyplot as plt

# Q-range scenarios (Othello max_utility=1, so Q ∈ [-1, 1])
SCENARIOS = [
    (2.0, "wide spread (Q_max=1, Q_min=-1)"),
    (1.0, "moderate spread"),
    (0.5, "narrow spread"),
    (0.1, "very tight spread"),
]

SCALE_FACTORS = [0.10, 0.20, 0.25, 0.35, 0.50]

# ---------------------------------------------------------------------------
fig, axes = plt.subplots(len(SCENARIOS), len(SCALE_FACTORS),
                         figsize=(len(SCALE_FACTORS) * 2.8, len(SCENARIOS) * 2.4))
fig.suptitle("Weak-move Boltzmann weight vs Q-distance to best\n"
             r"$w(\Delta Q) = \exp(-|\Delta Q| / T),\quad T = scale \times q\_range$",
             fontsize=12, fontweight="bold")

# Sample N hypothetical actions along the Q range
N_ACTIONS = 20

for row, (q_range, label) in enumerate(SCENARIOS):
    best_q = q_range * 0.8  # best Q near top of range
    # left=best Q, right=worst Q (descending Q)
    actions = np.linspace(best_q, best_q - q_range, N_ACTIONS)
    delta_q = best_q - actions  # 0 for best, q_range for worst

    for col, scale in enumerate(SCALE_FACTORS):
        ax = axes[row, col]
        T = max(q_range * scale, 0.01)
        weights = np.exp(-delta_q / T)
        probs = weights / weights.sum()

        colors = plt.cm.RdPu(0.3 + 0.7 * (delta_q / max(delta_q.max(), 0.001)))

        ax.bar(range(N_ACTIONS), probs, color=colors, edgecolor="white", linewidth=0.3)
        ax.set_title(f"scale={scale:.2f}  T={T:.3f}", fontsize=8)
        ax.set_xticks([0, N_ACTIONS - 1])
        ax.set_xticklabels([f"best\nQ={best_q:.1f}", f"worst\nQ={actions[-1]:.1f}"], fontsize=6)
        ax.tick_params(axis="y", labelsize=6)

        # Annotate best/worst prob ratio
        ratio = probs[0] / probs[-1] if probs[-1] > 0 else float("inf")
        ax.text(0.5, 0.92, f"best/worst={ratio:.1f}×", transform=ax.transAxes,
                fontsize=6, ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7))

        if col == 0:
            ax.set_ylabel(label + f"\n(q_range={q_range:.1f})", fontsize=7)

fig.tight_layout()
plt.show()
