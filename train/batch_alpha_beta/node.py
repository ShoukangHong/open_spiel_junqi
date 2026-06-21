"""Alpha-Beta search node — compatible with MCTS Node interface.

Provides the same attribute surface as train.batch_mcts.node.Node so
that compute_solved_policy() and play_game() work without changes.
"""

from typing import Optional

import numpy as np
import pyspiel


class AlphaBetaNode:
    """A node in the alpha-beta search tree.

    Mirrors the MCTS Node interface: action, player, prior, explore_count,
    total_reward, q_value, outcome, children, best_child().

    explore_count is 1 for visited children and 0 for pruned children.
    q_value == total_reward (since there is only one "visit").
    """

    __slots__ = (
        "action",
        "player",
        "prior",
        "explore_count",
        "total_reward",
        "draw_reward",
        "outcome",
        "children",
        "state",
    )

    def __init__(self, action: Optional[int], player: int, prior: float):
        self.action = action
        self.player = player
        self.prior = prior
        self.explore_count = 0
        self.total_reward = 0.0
        self.draw_reward = 0.0
        self.outcome = None
        self.children = []
        self.state = None

    @property
    def q_value(self) -> float:
        return self.total_reward  # explore_count is always 1 when visited

    @property
    def draw_rate(self) -> float:
        return self.draw_reward

    def sort_key(self) -> tuple:
        return (0 if self.outcome is None else self.outcome[self.player],
                self.explore_count, self.total_reward)

    def best_child(self) -> "AlphaBetaNode":
        return max(self.children, key=AlphaBetaNode.sort_key)

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
            for c in reversed(sorted(self.children, key=AlphaBetaNode.sort_key))
        )

    def __str__(self) -> str:
        return self.to_str(None)
