"""MCTS tree node — memory-optimized with virtual-loss support.

Key invariant: Q = total_reward / explore_count uses only REAL visits.
virtual_visits inflates N in PUCT to steer sibling threads away, but
never pollutes Q.
"""

import math
from typing import Optional

import pyspiel

from train.core.position_hash import hash_state


class Node:
    """A node in the MCTS search tree.

    Attributes:
        action: The action from the parent's perspective (None for root).
        player: The player who made `action` to reach this node.
            For root, equals current_player at root state.
            For non-root children, opposite of children[0].player (turns alternate).
        prior: Prior probability from the policy network.
        explore_count: Number of REAL visits (base truth for Q).
        total_reward: Sum of REAL returns (from `player`'s perspective) through this node.
        virtual_visits: Temporary visit inflation from in-flight threads.
        outcome: Terminal or proven outcome for all players (p0/p1 indexed), or None.
        children: Child Node instances. children[0].player is the player who moves
            FROM this node's state (= 1 - player for non-root, = player for root).
        drawable: Whether the player-to-move from this node (children[0].player)
            can achieve at least a draw — i.e. at least one child is a proven draw.
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
        "drawable",        # player-to-move (=children[0].player) can force ≥ draw
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
        self.drawable = False   # player-to-move can force ≥ draw
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
    def cur_player(self) -> int:
        """Player whose turn it is from this node's state.

        If children exist, this is children[0].player (the player who makes
        the next move).  Otherwise falls back to the root convention (root's
        player == current player) or the alternating-turn rule.
        """
        if self.children:
            return self.children[0].player
        # action is None only for root, where player already IS current player
        return self.player if self.action is None else 1 - self.player

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

        return math.atanh(self.q_value * 0.99999) + uct_c * self.prior * math.sqrt(
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
            q = max((1.0 + q) * (1.0 - repeat_penalty) - 1.0, -1)
        q = math.atanh(q * 0.99999)
        if self.explore_count == 0 and fpu_lambda > 0:
            q = q_parent - fpu_lambda * (prior_max - self.prior) / max(prior_max, 1e-9)
        return q + u

    # ── Reparent ──────────────────────────────────────────────────────────

    def reparent_as_root(self, scale: float = None,
                          recursive: bool = False) -> None:
        """Convert this node from a child into a new root, in-place.

        Flips player and negates total_reward to match the new perspective.
        If *scale* is given (e.g. 0.5), stats are shrunk so inherited data
        doesn't dominate the new search.

        With *recursive=False* (default): scales only direct children.
        With *recursive=True*: recurses into descendants, but only if the
        child's explore_count exceeds root threshold
        (sum of children's explore_count × scale).
        """
        self.total_reward = -self.total_reward
        self.player = 1 - self.player
        self.action = None
        self.noise_applied = False
        self._pos_hash = None
        self.virtual_visits = 0

        if scale is not None and scale != 1.0:
            root_n = sum(c.explore_count for c in self.children)
            self._scale_stats(scale)
            for c in self.children:
                if c.outcome is None:
                    if recursive:
                        c._reparent_scale_recursive(scale, root_n * scale)
                    else:
                        c._scale_stats(scale)

        # Children are now root-level; compute pos_hash so repeat penalty works.
        # Skip terminal children — solver excludes them from PUCT anyway.
        for c in self.children:
            if c._pos_hash is None and c.state is not None \
                    and not c.state.is_terminal():
                c._pos_hash = hash_state(c.state)

    def _scale_stats(self, scale: float) -> None:
        """Shrink explore_count by *scale*, then scale reward by the true
        ratio (new_N / old_N) to preserve Q exactly despite rounding."""
        old_n = self.explore_count
        self.explore_count = max(math.ceil(old_n * scale), 0)
        ratio = self.explore_count / max(old_n, 1)
        self.total_reward *= ratio
        self.draw_reward *= ratio

    def _reparent_scale_recursive(self, scale: float, threshold: float) -> None:
        """Scale this subtree.  Only self uses children's sum (may have
        been scaled); children's explore_count is still clean."""
        total_n = sum(c.explore_count for c in self.children) \
                  or self.explore_count
        if total_n <= threshold:
            return
        self._scale_stats(scale)
        for c in self.children:
            if c.outcome is None:
                if c.explore_count > threshold:
                    c._reparent_scale_recursive(scale, threshold)
                else:
                    c._scale_stats(scale)

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
