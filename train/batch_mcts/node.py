"""MCTS tree node — memory-optimized with virtual-loss support.

Key invariant: Q = total_reward / explore_count uses only REAL visits.
virtual_visits inflates N in PUCT to steer sibling threads away, but
never pollutes Q.
"""

import math
from typing import Optional

import pyspiel


class Node:
    """A node in the MCTS search tree.

    Attributes:
        action: The action from the parent's perspective (None for root).
        player: The player who made `action`.
        prior: Prior probability from the policy network.
        explore_count: Number of REAL visits (base truth for Q).
        total_reward: Sum of REAL rewards through this node.
        virtual_visits: Temporary visit inflation from in-flight threads.
        outcome: Terminal or proven outcome for all players, or None.
        children: Child Node instances.
    """
    draw_penalty: float = 0.0  # class-level: penalise draw-heavy branches in PUCT

    __slots__ = (
        "action",
        "player",
        "prior",
        "explore_count",
        "total_reward",
        "draw_reward",     # accumulated draw probability (WDL mode only)
        "virtual_visits",
        "outcome",
        "children",
        "state",           # cached state clone
        "noise_applied",   # whether Dirichlet noise has been applied
        "_pos_hash",       # hash of static board position (repeat detection)
        "nn_q",            # NN raw Q value (cached for surprise detection)
        "nn_draw",         # NN draw rate
        "nn_prior",        # NN prior distribution (list of (action, prob))
        "nn_prior_max",    # NN's max prior among legal actions
        "nn_argmax",       # NN's argmax action
        "drawable",        # at least one child is a proven draw
    )

    def __init__(self, action: Optional[int], player: int, prior: float):
        self.action = action
        self.player = player
        self.prior = prior
        self.explore_count = 0
        self.total_reward = 0.0
        self.draw_reward = 0.0
        self.virtual_visits = 0
        self.outcome = None
        self.drawable = False   # at least one child is a proven draw
        self.children = []
        self.state = None
        self.noise_applied = False
        self._pos_hash = None
        self.nn_q = None
        self.nn_draw = None
        self.nn_prior = None
        self.nn_prior_max = None
        self.nn_argmax = None

    # ── Q / N / PUCT ──────────────────────────────────────────────────────

    @property
    def visit_count(self) -> int:
        """Effective N including in-flight virtual visits."""
        return self.explore_count + self.virtual_visits

    @property
    def q_value(self) -> float:
        """Mean real reward (unbiased by virtual loss)."""
        if self.explore_count == 0:
            return 0.0
        return self.total_reward / self.explore_count

    @property
    def draw_rate(self) -> float:
        """Mean draw probability through this node (WDL mode)."""
        if self.explore_count == 0:
            return 0.0
        return self.draw_reward / self.explore_count

    def puct_value(self, parent_explore_count: int, uct_c: float) -> float:
        """Standard PUCT used by original MCTS (no virtual loss)."""
        if self.outcome is not None:
            return self.outcome[self.player]

        if self.explore_count == 0:
            return float("inf")

        return self.q_value + uct_c * self.prior * math.sqrt(
            parent_explore_count) / (self.explore_count + 1)

    def puct_with_virtual(self, parent_explore_count: int, uct_c: float,
                          virtual_loss: float, repeat_penalty: float = 0.0,
                          q_parent: float = 0.0, fpu_lambda: float = 0.0,
                          prior_max: float = 1.0) -> float:
        """PUCT with virtual loss, repeat penalty, and FPU.

        FPU: unvisited nodes get Q = Q_parent - λ·(p_max-p)/p_max.
        This focuses exploration on high-prior moves in losing positions.
        """
        if self.outcome is not None:
            return self.outcome[self.player]

        n = self.visit_count
        u = uct_c * self.prior * math.sqrt(max(parent_explore_count, 1)) / (n + 1)
        q = self.q_value - Node.draw_penalty * self.draw_rate
        if repeat_penalty > 0:
            q = (1.0 + q) * (1.0 - repeat_penalty) - 1.0
        if self.explore_count == 0 and fpu_lambda > 0:
            q = q_parent - fpu_lambda * (prior_max - self.prior) / max(prior_max, 1e-9)
        return q + u

    # ── Best child ────────────────────────────────────────────────────────

    def sort_key(self) -> tuple:
        """Sort key for best_child: proven > explored > expected reward."""
        return (0 if self.outcome is None else self.outcome[self.player],
                self.explore_count, self.total_reward)

    def best_child(self) -> "Node":
        return max(self.children, key=Node.sort_key)

    # ── String conversion ─────────────────────────────────────────────────

    def to_str(self, state: Optional[pyspiel.State] = None) -> str:
        action = (
            state.action_to_string(state.current_player(), self.action)
            if state and self.action is not None else str(self.action))
        return ("{:>6}: player: {}, prior: {:5.3f}, value: {:6.3f}, "
                "sims: {:5d}, outcome: {}, {:3d} children").format(
                    action, self.player, self.prior,
                    self.q_value if self.explore_count else 0.0,
                    self.explore_count,
                    ("{:4.1f}".format(self.outcome[self.player])
                     if self.outcome else "none"),
                    len(self.children))

    def children_str(self, state: Optional[pyspiel.State] = None) -> str:
        return "\n".join(
            c.to_str(state)
            for c in reversed(sorted(self.children, key=Node.sort_key))
        )

    def __str__(self) -> str:
        return self.to_str(None)
