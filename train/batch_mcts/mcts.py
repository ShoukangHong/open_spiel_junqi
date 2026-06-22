# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Monte-Carlo Tree Search algorithm for game play.

Contains:
  - Original MCTS (SearchNode, MCTSBot) — untouched from OpenSpiel
  - BatchMCTS — virtual loss + leaf batching for higher throughput
"""

import math
import time

import numpy as np

import pyspiel

from train.batch_mcts.node import Node
from train.batch_mcts.config import MCTSConfig
from train.core.position_hash import hash_state, hash_obs as _hash_obs


def compute_solved_policy(children, player, max_utility, alpha=5.0,
                          root_visits=None):
    """Solved-aware policy from root children.

    Confidence uses *root_visits* (not per-child) because solving stops
    individual exploration — only the parent accumulates more visits.

    Three cases:
      1. All children proven → best-outcome children split evenly.
      2. Some non-loss proven → reward/penalty: weight = exp(α×diff×√root_N).
      3. Only loss-proven or none → visit-proportional, loss proven zeroed.

    Returns: dict {action: probability}.
    """
    if root_visits is None:
        root_visits = max(sum(c.explore_count for c in children), 1)
    root_conf = math.sqrt(max(root_visits, 1))

    # Case 1: fully solved
    all_solved = all(c.outcome is not None for c in children)
    non_loss = [c for c in children
                if c.outcome is not None
                and c.outcome[player] > -max_utility]

    if all_solved:
        best_val = max(c.outcome[player] for c in children)
        best = [c for c in children if c.outcome[player] == best_val]
        if best_val >= 0:
            # Win or draw: prefer least-explored (cleanest path)
            weights = {c.action: 1.0 / max(c.explore_count, 1) for c in best}
        else:
            # Losing: prefer most-explored (most tested)
            weights = {c.action: float(c.explore_count) for c in best}
        total = sum(weights.values()) or 1.0
        return {c.action: (weights.get(c.action, 0.0) / total)
                for c in children}

    # Case 2: some non-loss proven → reward/penalty model
    if non_loss:
        eff = {}
        for c in children:
            if c.outcome is not None:
                eff[c.action] = c.outcome[player]
            elif c.explore_count > 0:
                eff[c.action] = c.q_value
        best_val = max(eff.values()) if eff else 0.0

        weights = {}
        for c in children:
            c_eff = eff.get(c.action, 0.0)
            if c.outcome is not None and c.outcome[player] == max_utility:
                diff = 1/math.sqrt(max(c.explore_count, 1))
            else:
                diff = c_eff - best_val
            conf = root_conf if c.outcome is not None else math.sqrt(max(c.explore_count, 1))
            weights[c.action] = math.exp(alpha * diff * conf)
        total_w = sum(weights.values())
        return {a: w / max(total_w, 1e-9) for a, w in weights.items()}

    # Case 3: only loss-proven or none
    loss_actions = {c.action for c in children
                    if c.outcome is not None
                    and c.outcome[player] < 0}
    total_visits = sum(c.explore_count for c in children
                       if c.action not in loss_actions)
    if total_visits > 0:
        return {c.action: (c.explore_count / total_visits
                           if c.action not in loss_actions else 0.0)
                for c in children}
    # No visits at all — use NN priors
    total_prior = sum(c.prior for c in children if c.action not in loss_actions)
    if total_prior > 0:
        return {c.action: (c.prior / total_prior
                           if c.action not in loss_actions else 0.0)
                for c in children}
    return {c.action: 1.0 / len(children) for c in children}


class Evaluator(object):
  """Abstract class representing an evaluation function for a game.

  The evaluation function takes in an intermediate state in the game and returns
  an evaluation of that state, which should correlate with chances of winning
  the game. It returns the evaluation from all player's perspectives.
  """

  def evaluate(self, state):
    """Returns evaluation on given state."""
    raise NotImplementedError

  def prior(self, state):
    """Returns a probability for each legal action in the given state."""
    raise NotImplementedError


class RandomRolloutEvaluator(Evaluator):
  """A simple evaluator doing random rollouts.

  This evaluator returns the average outcome of playing random actions from the
  given state until the end of the game.  n_rollouts is the number of random
  outcomes to be considered.
  """

  def __init__(self, n_rollouts=1, random_state=None, max_length=None):
    self.n_rollouts = n_rollouts
    self.max_length = max_length
    self._random_state = random_state or np.random.RandomState()

  def evaluate(self, state):
    """Returns evaluation on given state."""
    result = None
    for _ in range(self.n_rollouts):
      working_state = state.clone()
      length = 0
      while not working_state.is_terminal():
        if working_state.is_chance_node():
          outcomes = working_state.chance_outcomes()
          action_list, prob_list = zip(*outcomes)
          action = self._random_state.choice(action_list, p=prob_list)
        else:
          action = self._random_state.choice(working_state.legal_actions())
        working_state.apply_action(action)
        length += 1
        if self.max_length is not None and length >= self.max_length:
          break
      returns = np.array(working_state.returns())
      result = returns if result is None else result + returns

    return result / self.n_rollouts

  def prior(self, state):
    """Returns equal probability for all actions."""
    if state.is_chance_node():
      return state.chance_outcomes()
    else:
      legal_actions = state.legal_actions(state.current_player())
      return [(action, 1.0 / len(legal_actions)) for action in legal_actions]


class SearchNode(object):
  """A node in the search tree.

  A SearchNode represents a state and possible continuations from it. Each child
  represents a possible action, and the expected result from doing so.

  Attributes:
    action: The action from the parent node's perspective. Not important for the
      root node, as the actions that lead to it are in the past.
    player: Which player made this action.
    prior: A prior probability for how likely this action will be selected.
    explore_count: How many times this node was explored.
    total_reward: The sum of rewards of rollouts through this node, from the
      parent node's perspective. The average reward of this node is
      `total_reward / explore_count`
    outcome: The rewards for all players if this is a terminal node or the
      subtree has been proven, otherwise None.
    children: A list of SearchNodes representing the possible actions from this
      node, along with their expected rewards.
  """
  __slots__ = [
      "action",
      "player",
      "prior",
      "explore_count",
      "total_reward",
      "outcome",
      "children",
  ]

  def __init__(self, action, player, prior):
    self.action = action
    self.player = player
    self.prior = prior
    self.explore_count = 0
    self.total_reward = 0.0
    self.outcome = None
    self.children = []

  def uct_value(self, parent_explore_count, uct_c):
    """Returns the UCT value of child."""
    if self.outcome is not None:
      return self.outcome[self.player]

    if self.explore_count == 0:
      return float("inf")

    return self.total_reward / self.explore_count + uct_c * math.sqrt(
        math.log(parent_explore_count) / self.explore_count)

  def puct_value(self, parent_explore_count, uct_c):
    """Returns the PUCT value of child."""
    if self.outcome is not None:
      return self.outcome[self.player]

    return ((self.explore_count and self.total_reward / self.explore_count) +
            uct_c * self.prior * math.sqrt(parent_explore_count) /
            (self.explore_count + 1))

  def sort_key(self):
    """Returns the best action from this node, either proven or most visited.

    This ordering leads to choosing:
    - Highest proven score > 0 over anything else, including a promising but
      unproven action.
    - A proven draw only if it has higher exploration than others that are
      uncertain, or the others are losses.
    - Uncertain action with most exploration over loss of any difficulty
    - Hardest loss if everything is a loss
    - Highest expected reward if explore counts are equal (unlikely).
    - Longest win, if multiple are proven (unlikely due to early stopping).
    """
    return (0 if self.outcome is None else self.outcome[self.player],
            self.explore_count, self.total_reward)

  def best_child(self):
    """Returns the best child in order of the sort key."""
    return max(self.children, key=SearchNode.sort_key)

  def children_str(self, state=None):
    """Returns the string representation of this node's children.

    They are ordered based on the sort key, so order of being chosen to play.

    Args:
      state: A `pyspiel.State` object, to be used to convert the action id into
        a human readable format. If None, the action integer id is used.
    """
    return "\n".join([
        c.to_str(state)
        for c in reversed(sorted(self.children, key=SearchNode.sort_key))
    ])

  def to_str(self, state=None):
    """Returns the string representation of this node.

    Args:
      state: A `pyspiel.State` object, to be used to convert the action id into
        a human readable format. If None, the action integer id is used.
    """
    action = (
        state.action_to_string(state.current_player(), self.action)
        if state and self.action is not None else str(self.action))
    return ("{:>6}: player: {}, prior: {:5.3f}, value: {:6.3f}, sims: {:5d}, "
            "outcome: {}, {:3d} children").format(
                action, self.player, self.prior, self.explore_count and
                self.total_reward / self.explore_count, self.explore_count,
                ("{:4.1f}".format(self.outcome[self.player])
                 if self.outcome else "none"), len(self.children))

  def __str__(self):
    return self.to_str(None)


class MCTSBot(pyspiel.Bot):
  """Bot that uses Monte-Carlo Tree Search algorithm."""

  def __init__(self,
               game,
               uct_c,
               max_simulations,
               evaluator,
               solve=True,
               random_state=None,
               child_selection_fn=SearchNode.uct_value,
               dirichlet_noise=None,
               verbose=False,
               dont_return_chance_node=False):
    """Initializes a MCTS Search algorithm in the form of a bot.

    In multiplayer games, or non-zero-sum games, the players will play the
    greedy strategy.

    Args:
      game: A pyspiel.Game to play.
      uct_c: The exploration constant for UCT.
      max_simulations: How many iterations of MCTS to perform. Each simulation
        will result in one call to the evaluator. Memory usage should grow
        linearly with simulations * branching factor. How many nodes in the
        search tree should be evaluated. This is correlated with memory size and
        tree depth.
      evaluator: A `Evaluator` object to use to evaluate a leaf node.
      solve: Whether to back up solved states.
      random_state: An optional numpy RandomState to make it deterministic.
      child_selection_fn: A function to select the child in the descent phase.
        The default is UCT.
      dirichlet_noise: A tuple of (epsilon, alpha) for adding dirichlet noise to
        the policy at the root. This is from the alpha-zero paper.
      verbose: Whether to print information about the search tree before
        returning the action. Useful for confirming the search is working
        sensibly.
      dont_return_chance_node: If true, do not stop expanding at chance nodes.
        Enabled for AlphaZero.

    Raises:
      ValueError: if the game type isn't supported.
    """
    pyspiel.Bot.__init__(self)
    # Check that the game satisfies the conditions for this MCTS implemention.
    game_type = game.get_type()
    if game_type.reward_model != pyspiel.GameType.RewardModel.TERMINAL:
      raise ValueError("Game must have terminal rewards.")
    if game_type.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
      raise ValueError("Game must have sequential turns.")

    self._game = game
    self.uct_c = uct_c
    self.max_simulations = max_simulations
    self.evaluator = evaluator
    self.verbose = verbose
    self.solve = solve
    self.max_utility = game.max_utility()
    self._dirichlet_noise = dirichlet_noise
    self._random_state = random_state or np.random.RandomState()
    self._child_selection_fn = child_selection_fn
    self.dont_return_chance_node = dont_return_chance_node

  def restart_at(self, state):
    pass

  def step_with_policy(self, state):
    """Returns bot's policy and action at given state.

    Returns an invalid action policy and action if the state is a chance node.

    Args:
      state: pyspiel.State object, state to search from

    Returns:
      policy: A list of (action, probability) pairs.
      action: The action the bot takes.
    """
    if state.is_chance_node():
      print("Chance node, returning invalid action policy.")
      policy = [(pyspiel.INVALID_ACTION, 1.0)]
      return policy, pyspiel.INVALID_ACTION

    t1 = time.time()
    root = self.mcts_search(state)

    best = root.best_child()

    if self.verbose:
      seconds = time.time() - t1
      print("Finished {} sims in {:.3f} secs, {:.1f} sims/s".format(
          root.explore_count, seconds, root.explore_count / seconds))
      print("Root:")
      print(root.to_str(state))
      print("Children:")
      print(root.children_str(state))
      if best.children:
        chosen_state = state.clone()
        chosen_state.apply_action(best.action)
        print("Children of chosen:")
        print(best.children_str(chosen_state))

    mcts_action = best.action

    policy = [(action, (1.0 if action == mcts_action else 0.0))
              for action in state.legal_actions(state.current_player())]

    return policy, mcts_action

  def step(self, state):
    return self.step_with_policy(state)[1]

  def _apply_tree_policy(self, root, state):
    """Applies the UCT policy to play the game until reaching a leaf node.

    A leaf node is defined as a node that is terminal or has not been evaluated
    yet. If it reaches a node that has been evaluated before but hasn't been
    expanded, then expand it's children and continue.

    Args:
      root: The root node in the search tree.
      state: The state of the game at the root node.

    Returns:
      visit_path: A list of nodes descending from the root node to a leaf node.
      working_state: The state of the game at the leaf node.
    """
    visit_path = [root]
    working_state = state.clone()
    current_node = root
    while (not working_state.is_terminal() and
           current_node.explore_count > 0) or (
               working_state.is_chance_node() and self.dont_return_chance_node):
      if not current_node.children:
        # For a new node, initialize its state, then choose a child as normal.
        legal_actions = self.evaluator.prior(working_state)
        if current_node is root and self._dirichlet_noise:
          epsilon, alpha = self._dirichlet_noise
          noise = self._random_state.dirichlet([alpha] * len(legal_actions))
          legal_actions = [(a, (1 - epsilon) * p + epsilon * n)
                           for (a, p), n in zip(legal_actions, noise)]
        # Reduce bias from move generation order.
        self._random_state.shuffle(legal_actions)
        player = working_state.current_player()
        current_node.children = [
            SearchNode(action, player, prior) for action, prior in legal_actions
        ]

      if working_state.is_chance_node():
        # For chance nodes, rollout according to chance node's probability
        # distribution
        outcomes = working_state.chance_outcomes()
        action_list, prob_list = zip(*outcomes)
        action = self._random_state.choice(action_list, p=prob_list)
        chosen_child = next(
            c for c in current_node.children if c.action == action)
      else:
        # Otherwise choose node with largest UCT value
        chosen_child = max(
            current_node.children,
            key=lambda c: self._child_selection_fn(  # pylint: disable=g-long-lambda
                c, current_node.explore_count, self.uct_c))

      working_state.apply_action(chosen_child.action)
      current_node = chosen_child
      visit_path.append(current_node)

    return visit_path, working_state

  def mcts_search(self, state):
    """A vanilla Monte-Carlo Tree Search algorithm.

    This algorithm searches the game tree from the given state.
    At the leaf, the evaluator is called if the game state is not terminal.
    A total of max_simulations states are explored.

    At every node, the algorithm chooses the action with the highest PUCT value,
    defined as: `Q/N + c * prior * sqrt(parent_N) / N`, where Q is the total
    reward after the action, and N is the number of times the action was
    explored in this position. The input parameter c controls the balance
    between exploration and exploitation; higher values of c encourage
    exploration of under-explored nodes. Unseen actions are always explored
    first.

    At the end of the search, the chosen action is the action that has been
    explored most often. This is the action that is returned.

    This implementation supports sequential n-player games, with or without
    chance nodes. All players maximize their own reward and ignore the other
    players' rewards. This corresponds to max^n for n-player games. It is the
    norm for zero-sum games, but doesn't have any special handling for
    non-zero-sum games. It doesn't have any special handling for imperfect
    information games.

    The implementation also supports backing up solved states, i.e. MCTS-Solver.
    The implementation is general in that it is based on a max^n backup (each
    player greedily chooses their maximum among proven children values, or there
    exists one child whose proven value is game.max_utility()), so it will work
    for multiplayer, general-sum, and arbitrary payoff games (not just win/loss/
    draw games). Also chance nodes are considered proven only if all children
    have the same value.

    Some references:
    - Sturtevant, An Analysis of UCT in Multi-Player Games,  2008,
      https://web.cs.du.edu/~sturtevant/papers/multi-player_UCT.pdf
    - Nijssen, Monte-Carlo Tree Search for Multi-Player Games, 2013,
      https://project.dke.maastrichtuniversity.nl/games/files/phd/Nijssen_thesis.pdf
    - Silver, AlphaGo Zero: Starting from scratch, 2017
      https://deepmind.com/blog/article/alphago-zero-starting-scratch
    - Winands, Bjornsson, and Saito, "Monte-Carlo Tree Search Solver", 2008.
      https://dke.maastrichtuniversity.nl/m.winands/documents/uctloa.pdf

    Arguments:
      state: pyspiel.State object, state to search from

    Returns:
      The most visited move from the root node.
    """
    root = SearchNode(None, state.current_player(), 1)
    for _ in range(self.max_simulations):
      visit_path, working_state = self._apply_tree_policy(root, state)
      if working_state.is_terminal():
        returns = working_state.returns()
        visit_path[-1].outcome = returns
        solved = self.solve
      else:
        returns = self.evaluator.evaluate(working_state)
        solved = False

      while visit_path:
        # For chance nodes, walk up the tree to find the decision-maker.
        decision_node_idx = -1
        while visit_path[decision_node_idx].player == pyspiel.PlayerId.CHANCE:
          decision_node_idx -= 1
        # Chance node targets are for the respective decision-maker.
        target_return = returns[visit_path[decision_node_idx].player]
        node = visit_path.pop()
        node.total_reward += target_return
        node.explore_count += 1

        if solved and node.children:
          player = node.children[0].player
          if player == pyspiel.PlayerId.CHANCE:
            # Only back up chance nodes if all have the same outcome.
            # An alternative would be to back up the weighted average of
            # outcomes if all children are solved, but that is less clear.
            outcome = node.children[0].outcome
            if (outcome is not None and
                all(np.array_equal(c.outcome, outcome) for c in node.children)):
              node.outcome = outcome
            else:
              solved = False
          else:
            # If any have max utility (won?), or all children are solved,
            # choose the one best for the player choosing.
            best = None
            all_solved = True
            for child in node.children:
              if child.outcome is None:
                all_solved = False
              elif best is None or child.outcome[player] > best.outcome[player]:
                best = child
            if (best is not None and
                (all_solved or best.outcome[player] == self.max_utility)):
              node.outcome = best.outcome
            else:
              solved = False
      if root.outcome is not None:
        break

    return root


# ═══════════════════════════════════════════════════════════════════════════
#  BatchMCTS — virtual loss + leaf batching
# ═══════════════════════════════════════════════════════════════════════════

class BatchMCTS:
    """Batch Monte Carlo Tree Search with virtual loss.

    Instead of running simulations one-by-one, this runs them in batches:
      1. N virtual threads traverse the tree concurrently using virtual loss
         to avoid collisions.
      2. All leaf states are batch-evaluated through the neural network.
      3. Results are backpropagated, removing virtual losses.

    Key invariant: virtual_visits affects PUCT exploration (inflates N) but
    NEVER pollutes Q = total_reward / explore_count.
    """

    def __init__(self, game, config=None, evaluator=None,
                 random_state=None):
        """Initialize BatchMCTS.

        Args:
            game: A pyspiel.Game.
            config: MCTSConfig instance. Uses defaults if None.
            evaluator: A BatchEvaluator-compatible object.
            random_state: Optional numpy RandomState.
        """
        game_type = game.get_type()
        if game_type.reward_model != pyspiel.GameType.RewardModel.TERMINAL:
            raise ValueError("Game must have terminal rewards.")
        if game_type.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
            raise ValueError("Game must have sequential turns.")

        self._game = game
        self.config = config or MCTSConfig()
        Node.draw_penalty = self.config.draw_penalty
        self.evaluator = evaluator
        self.max_utility = game.max_utility()
        self._random_state = random_state or np.random.RandomState()
        self._rep_counts = {}
        self._repeat_penalty = self.config.repeat_penalty

    # ── Public API ──────────────────────────────────────────────────────

    def mcts_search(self, state, root=None):
        """Run batch MCTS from `state`, returning the root Node.

        If *root* is given, simulations are ADDED to the existing tree
        (persistent search).  Otherwise a fresh tree is created.
        """
        if root is None:
            root = Node(None, state.current_player(), 1)
            root.state = state.clone()

        # Already fully solved (persistent search) — nothing to do
        if root.children and all(c.outcome is not None
                                 for c in root.children):
            return root

        # ── Build repetition count from game history ─────────────────────
        tmp = self._game.new_initial_state()
        rep_counts = {}  # position_hash → occurrence count
        if not tmp.is_terminal():
            rep_counts[hash_state(tmp)] = 1
        for a in state.history():
            tmp.apply_action(a)
            if not tmp.is_terminal():
                h = hash_state(tmp)
                rep_counts[h] = rep_counts.get(h, 0) + 1

        self._rep_counts = rep_counts
        self._repeat_penalty = self.config.repeat_penalty  # per-occurrence, capped

        # ── Speculative probe: pre-expand along NN-prior-best path ──────
        if self.config.probe_depth > 0 and not root.children:
            self._speculative_probe(root, state)

        max_sim = self.config.max_simulations
        batch_size = self.config.batch_size

        sims_done = 0
        while sims_done < max_sim:
            remaining = max_sim - sims_done
            batch_limit = min(batch_size, remaining)

            paths, early_flush = self._collect_paths_plain(
                root, state, batch_limit)

            if paths:
                sims_done += self._flush_paths(root, paths)

            if root.outcome is not None:
                break
            if root.children and all(c.outcome is not None
                                     for c in root.children):
                break
            if not early_flush and len(paths) < batch_limit:
                break  # all leaves already explored, can't progress

        return root

    def _flush_paths(self, root, paths):
        """Run Phase 2+3 on *paths*, return number of paths processed."""
        # ── Phase 2: Batch evaluate ──────────────────────────────────
        unique_nodes = []
        node_to_idx = {}
        for _, leaf_node, leaf_state in paths:
            if leaf_node not in node_to_idx and not leaf_state.is_terminal():
                node_to_idx[leaf_node] = len(unique_nodes)
                unique_nodes.append(leaf_node)

        values_map = {}
        if unique_nodes:
            states_to_eval = [n.state for n in unique_nodes]
            values_arr, priors_list = self.evaluator.batch_inference_raw(
                states_to_eval)
            for node, out, prior in zip(unique_nodes, values_arr,
                                        priors_list):
                w, d, l = float(out[0]), float(out[1]), float(out[2])
                value = (w - l) * self.max_utility
                values_map[node] = (value, prior, d)

        # ── Phase 3: Expand + Backprop ───────────────────────────────
        expanded_this_batch = set()
        for path_nodes, leaf_node, leaf_state in paths:
            if leaf_state.is_terminal():
                returns = np.array(leaf_state.returns())
                draw_prob = 1.0 if all(r == 0 for r in returns) else 0.0
            else:
                value, prior, draw_prob = values_map[leaf_node]
                if leaf_node not in expanded_this_batch:
                    self._expand(leaf_node, leaf_state, prior)
                    expanded_this_batch.add(leaf_node)
                lp = leaf_state.current_player()
                returns = np.zeros(2)
                returns[lp] = value
                returns[1 - lp] = -value

            self._backprop(path_nodes, returns, draw_prob)

            if leaf_state.is_terminal():
                leaf_node.outcome = returns

            for node in reversed(path_nodes):
                if self._check_solved(node):
                    if node is root:
                        break

        return len(unique_nodes)

    def _collect_paths_plain(self, root, state, batch_limit):
        """Collect *batch_limit* paths without dedup filtering.

        Returns (paths, early_flush).  early_flush is always False — all
        paths are kept, so len(paths) < batch_limit never happens.
        """
        paths = []
        for _ in range(batch_limit):
            pn, ln, ls = self._select(root, state)
            paths.append((pn, ln, ls))
        return paths, False

    def _speculative_probe(self, root, state):
        """Pre-expand along each root action's NN-prior-best line.

        For each root action, follows the max-prior child for up to
        *probe_depth* layers.  If the NN value drops by more than
        *probe_surprise* relative to the parent, the line is terminated
        early (value trap detected).  All evaluated nodes are backpropped
        into the tree so subsequent MCTS search benefits.
        """
        config = self.config
        max_depth = config.probe_depth
        threshold = config.probe_surprise
        if max_depth <= 0 or self.evaluator is None:
            return

        legal = state.legal_actions()
        if not legal:
            return
        player = state.current_player()
        max_u = self.max_utility

        def _max_prior_action(s, prior_list):
            """Pick the legal action with highest prior."""
            best_a = None
            best_p = -1.0
            for a, p in prior_list:
                if p > best_p:
                    best_p = p
                    best_a = a
            if best_a is None:
                best_a = s.legal_actions()[0]
            return best_a

        # ── Level 0: eval root+children, expand root children ──────────
        root_states = []
        for a in legal:
            cs = state.clone()
            cs.apply_action(a)
            root_states.append(cs)
        to_eval = [state] + root_states
        vals0, priors0 = self.evaluator.batch_inference_raw(to_eval)
        root_priors = priors0[0]
        child_priors0 = priors0[1:]

        # action → (state, wdl, priors)
        _act_map = {}
        for i, a in enumerate(legal):
            _act_map[a] = (root_states[i], vals0[1 + i], child_priors0[i])

        self._expand(root, state, root_priors)
        for c in root.children:
            if c.action in _act_map:
                c.state = _act_map[c.action][0]

        # Paths: (node_list, state, prior_list, root_child_Q).  No backprop yet.
        paths = []
        for c in root.children:
            a = c.action
            if a not in _act_map or c.state.is_terminal():
                continue
            wdl_root_c = _act_map[a][1]
            q0 = float(wdl_root_c[0] - wdl_root_c[2]) * max_u
            eval_p0 = c.state.current_player()
            q0_root = -q0 if eval_p0 != player else q0
            paths.append(([root, c], c.state, _act_map[a][2], q0_root))

        # ── Walk deeper ───────────────────────────────────────────────
        for d in range(1, max_depth):
            if not paths:
                break

            next_states = []
            next_info = []   # (path_idx, next_state, action)
            next_term = []
            for p_idx, (bp_path, cs, pr, q0) in enumerate(paths):
                if cs.is_terminal():
                    continue
                best_a = _max_prior_action(cs, pr)
                ns = cs.clone()
                ns.apply_action(best_a)
                if ns.is_terminal():
                    next_term.append((p_idx, ns, best_a))
                else:
                    next_states.append(ns)
                    next_info.append((p_idx, ns, best_a))

            if next_states:
                vals_d, priors_d = self.evaluator.batch_inference_raw(next_states)

            new_paths = []
            for j, (p_idx, ns, best_a) in enumerate(next_info):
                wdl = vals_d[j]
                q_nn = float(wdl[0] - wdl[2]) * max_u
                eval_p = ns.current_player()
                q = -q_nn if eval_p != player else q_nn
                bp_path = paths[p_idx][0]
                _, _, _, q0 = paths[p_idx]  # root child Q for surprise
                parent = bp_path[-1]
                child_node = Node(best_a, 1 - parent.player,
                                  dict(paths[p_idx][2]).get(best_a, 0.01))
                child_node.state = ns
                # Check surprise
                if abs(q0 - q) > threshold * max_u:
                    # Surprise — backprop NOW and terminate this path
                    ret2 = np.zeros(2)
                    ret2[player] = q
                    ret2[1 - player] = -q
                    for _ in range(max_depth):
                        self._backprop(bp_path + [child_node], ret2)
                    continue  # don't add to new_paths
                new_paths.append((bp_path + [child_node], ns, priors_d[j], q0))
            # Terminal — backprop NOW (only backprop once at leaf/term)
            for (p_idx, ns, best_a) in next_term:
                bp_path = paths[p_idx][0]
                parent = bp_path[-1]
                child_node = Node(best_a, 1 - parent.player,
                                  dict(paths[p_idx][2]).get(best_a, 0.01))
                child_node.state = ns
                child_node.outcome = np.array(ns.returns(), dtype=np.float64)
                ret = ns.returns()
                # Convert returns to root perspective
                q = ret[0] * max_u  # p0 perspective
                if player != 0:
                    q = -q
                ret2 = np.zeros(2)
                ret2[player] = q
                ret2[1 - player] = -q
                for _ in range(max_depth):
                    self._backprop(bp_path + [child_node], ret2)
            paths = new_paths

        # ── At final depth: batch-eval leaves, backprop once per path ─
        if paths:
            leaf_states = []
            leaf_info = []
            for bp_path, ns, pr, q0 in paths:
                if ns.is_terminal():
                    leaf_info.append((True, ns.returns()))
                else:
                    leaf_states.append(ns)
                    leaf_info.append((False, len(leaf_states) - 1))

            if leaf_states:
                leaf_vals, _ = self.evaluator.batch_inference_raw(leaf_states)

            leaf_idx = 0
            for p_idx, (bp_path, ns, pr, q0) in enumerate(paths):
                is_term, info = leaf_info[p_idx]
                if is_term:
                    q = info[0] * max_u  # p0 perspective
                    if player != 0:
                        q = -q
                else:
                    wdl = leaf_vals[info]
                    q = float(wdl[0] - wdl[2]) * max_u
                    eval_p = ns.current_player()
                    if eval_p != player:
                        q = -q
                ret2 = np.zeros(2)
                ret2[player] = q
                ret2[1 - player] = -q
                for _ in range(max_depth):
                    self._backprop(bp_path, ret2)


    def _collect_paths_dedup(self, root, state, batch_limit):
        """Collect unique-leaf paths (no internal flush).
        responsible for flushing the returned paths.

        Returns (paths, early_flush).  early_flush is True when duplicates
        were skipped, signalling the caller that len(paths) < batch_limit
        is NOT due to tree exhaustion.
        """
        paths = []
        seen = set()
        early_flush = False
        for _ in range(batch_limit):
            pn, ln, ls = self._select(root, state)
            if ln in seen:
                for n in pn:
                    if n.virtual_visits > 0:
                        n.virtual_visits -= 1
                early_flush = True
                continue
            seen.add(ln)
            paths.append((pn, ln, ls))
        return paths, early_flush

    def compute_root_policy(self, root, state=None):
        """Return {action: probability} from MCTS visit distribution."""
        player = root.children[0].player if root.children else 0
        return compute_solved_policy(
            root.children, player, self.max_utility,
            root_visits=root.explore_count)

    def step(self, state):
        """Return the best action from the given state."""
        return self.step_with_policy(state)[1]

    def step_with_policy(self, state, temperature=0.0):
        """Return (policy, action) for the given state.

        Policy is visit-count proportional unless the solver has proven
        winning children, in which case all mass goes to those children.
        If *temperature* > 0, the action is sampled from the policy
        sharpened by τ (τ=0 → greedy, τ=1 → raw distribution).
        """
        if state.is_chance_node():
            return [(pyspiel.INVALID_ACTION, 1.0)], pyspiel.INVALID_ACTION

        t1 = time.time()
        root = self.mcts_search(state)
        self._last_root = root
        best = root.best_child()

        if self.config.verbose:
            print("Root:")
            print(root.to_str(state))
            print("Children:")
            print(root.children_str(state))

        # Solved-aware policy.
        player = state.current_player()
        policy_dict = compute_solved_policy(
            root.children, player, self.max_utility,
            root_visits=root.explore_count)
        policy = [(a, policy_dict.get(a, 0.0))
                  for a in state.legal_actions(state.current_player())]

        if temperature > 0 and len(policy) > 1:
            actions, probs = zip(*policy)
            probs = np.array(probs, dtype=np.float64)
            probs = probs ** (1.0 / temperature)
            probs /= probs.sum()
            action = actions[self._random_state.choice(len(actions), p=probs)]
        else:
            action = best.action
        return policy, action

    # ── Internal methods ───────────────────────────────────────────────

    def _select(self, root, init_state):
        """Traverse from *root* using PUCT+virtual_loss to reach a leaf.

        A leaf is a node with explore_count==0 (unexpanded) or a terminal
        state. Virtual loss is applied to every child selected along the
        way so that other threads in the same batch are steered elsewhere.

        Returns (path_nodes, leaf_node, leaf_state).
        """
        path = [root]
        node = root
        state = init_state.clone()

        while node.explore_count > 0 and not state.is_terminal():
            if (state.is_chance_node()
                    and node.children):
                # For chance nodes, sample according to probabilities
                outcomes = state.chance_outcomes()
                action_list, prob_list = zip(*outcomes)
                action = self._random_state.choice(action_list, p=prob_list)
                node = next(c for c in node.children if c.action == action)
                state.apply_action(action)
                path.append(node)
                continue

            if not node.children:
                break  # unexpanded leaf — stop here

            # Dirichlet noise at root (AlphaZero) — apply ONCE
            if node is root and not node.noise_applied and self.config.policy_epsilon:
                epsilon = self.config.policy_epsilon
                alpha = self.config.policy_alpha
                noise = self._random_state.dirichlet(
                    [alpha] * len(node.children))
                for child, n in zip(node.children, noise):
                    child.prior = ((1 - epsilon) * child.prior
                                   + epsilon * n)
                node.noise_applied = True

            # Select child with virtual-loss-adjusted PUCT.
            # Prune children proven worse than another sibling.
            uct_c = self.config.uct_c
            vloss = self.config.virtual_loss
            candidates = node.children
            if self.config.solve:
                player = node.children[0].player
                best_proven = -float("inf")
                for c in node.children:
                    if c.outcome is not None and c.outcome[player] > best_proven:
                        best_proven = c.outcome[player]
                if best_proven > -float("inf"):
                    # Exclude proven children — their value is settled.
                    # If a win is already proven, exploring other children
                    # is harmless (solve-aware policy handles the final
                    # output) and more informative.
                    candidates = [c for c in node.children
                                  if c.outcome is None]
                    if not candidates:
                        candidates = [c for c in node.children
                                      if c.outcome is not None
                                      and c.outcome[player] >= best_proven]
            # Unvisited children of root get a large bonus so the first
            # batches naturally cover every legal action — none skipped.
            # Repeated positions get a multiplicative Q penalty:
            #   Q' = (1+Q)*(1-penalty)-1
            # This scales the penalty with Q — strong positions are
            # penalised more, favouring the passive player.
            def _repeat_pen(c):
                if node is not root or c._pos_hash is None:
                    return 0.0
                count = self._rep_counts.get(c._pos_hash, 0)
                return min(count * self._repeat_penalty, 0.8)
            # FPU: unvisited nodes inherit parent Q minus prior-based penalty
            _fpu_lambda = self.config.fpu_lambda
            _q_parent = node.q_value
            _p_max = max((c.prior for c in candidates), default=1.0)
            best_child = max(
                candidates,
                key=lambda c: c.puct_with_virtual(
                    node.visit_count, uct_c, vloss, _repeat_pen(c),
                    q_parent=_q_parent, fpu_lambda=_fpu_lambda,
                    prior_max=_p_max)
                + (1e6 if node is root and c.explore_count == 0 else 0))

            # Apply virtual loss
            best_child.virtual_visits += 1

            if best_child.state is not None:
                state = best_child.state.clone()
            else:
                state.apply_action(best_child.action)
            node = best_child
            path.append(node)

        # Lazy: only set state for the leaf (needed by Phase 2 inference)
        if node.state is None:
            node.state = state.clone()
        return path, node, state

    def _expand(self, node, state, prior):
        """Create children for *node* from the prior probabilities.
        Child states are created lazily when _select reaches a leaf.
        """
        player = state.current_player()
        self._random_state.shuffle(prior)
        children = []
        for action, prob in prior:
            child = Node(action, player, prob)
            if node.action is None:  # root: pre-compute position hash + cache state
                s = state.clone()
                s.apply_action(action)
                if not s.is_terminal():
                    child._pos_hash = hash_state(s)
                child.state = s
            children.append(child)
        node.children = children

    def _backprop(self, path, returns, draw_prob=0.0):
        """Backpropagate *returns* and *draw_prob* along *path*.

        Skips nodes that already have a proven outcome — their value is
        settled and extra visits only inflate counts.
        """
        for i in range(len(path) - 1, -1, -1):
            node = path[i]
            # Always clean up virtual loss, even for proven nodes
            if node.virtual_visits > 0:
                node.virtual_visits -= 1
            if node.outcome is not None:
                continue
            decision_idx = i
            while path[decision_idx].player == pyspiel.PlayerId.CHANCE:
                decision_idx -= 1
            target = returns[path[decision_idx].player]

            node.total_reward += target
            node.draw_reward += draw_prob
            node.explore_count += 1

    def _check_solved(self, node):
        """Attempt to prove *node* using MCTS-Solver logic.

        Returns True if the node was proven (outcome set).
        """
        if not node.children:
            return False
        player = node.children[0].player
        if player == pyspiel.PlayerId.CHANCE:
            outcome = node.children[0].outcome
            if outcome is not None and all(
                    np.array_equal(c.outcome, outcome) for c in node.children):
                node.outcome = outcome
                self._clamp_draw(node)
                return True
        else:
            best_child = None
            all_solved = True
            for child in node.children:
                if child.outcome is None:
                    all_solved = False
                elif (best_child is None
                      or child.outcome[player] > best_child.outcome[player]):
                    best_child = child
            if best_child is not None and (
                    all_solved or best_child.outcome[player] == self.max_utility):
                node.outcome = best_child.outcome
                self._clamp_draw(node)
                return True
        return False

    def _clamp_draw(self, node):
        if all(r == 0 for r in node.outcome):
            node.draw_reward = float(node.explore_count)
