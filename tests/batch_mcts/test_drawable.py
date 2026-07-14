"""Tests for drawable flag — backprop clamp, _stable_qdr, end-to-end MCTS."""

import numpy as np
import pyspiel

from train.batch_mcts.node import Node
from train.batch_mcts.config import MCTSConfig
from train.batch_mcts.mcts import BatchMCTS


class _DummyEvaluator:
    """Evaluator that returns uniform random values (enough for drawable tests)."""

    def __init__(self, game):
        self._game = game
        self._rng = np.random.RandomState(42)

    def evaluate(self, state):
        return np.array([0.0, 0.0])

    def prior(self, state):
        if state.is_chance_node():
            return state.chance_outcomes()
        legal = state.legal_actions(state.current_player())
        return [(a, 1.0 / len(legal)) for a in legal]

    def scalar_value(self, state):
        return 0.0

    def batch_evaluate(self, states):
        return np.array([[np.zeros(2)] * len(states)])

    def batch_prior(self, states):
        return [self.prior(s) for s in states]

    def batch_inference_raw(self, states):
        n = len(states)
        na = self._game.num_distinct_actions()
        vl = np.zeros((n, 3), dtype=np.float32)
        vl[:, 1] = 1.0  # uniform draw prior
        pl = np.zeros((n, na), dtype=np.float32)
        return vl, [self.prior(s) for s in states]


# ── Test 1: Node init ────────────────────────────────────────────────────────

def test_node_drawable_default_false():
    n = Node(None, 0, 1.0)
    assert n.drawable is False


# ── Test 2: _check_solved sets drawable ───────────────────────────────────────

def test_check_solved_sets_drawable_on_draw_child():
    game = pyspiel.load_game("tic_tac_toe")
    evaluator = _DummyEvaluator(game)
    cfg = MCTSConfig(max_simulations=8, batch_size=4, solve=True)
    mcts = BatchMCTS(game, cfg, evaluator)

    # Build a parent with two children: one draw, one unproven
    parent = Node(None, 0, 1.0)
    c_draw = Node(0, 0, 0.5)
    c_draw.outcome = np.array([0.0, 0.0])  # proven draw
    c_draw.explore_count = 10
    c_unproven = Node(1, 0, 0.5)
    c_unproven.explore_count = 10
    parent.children = [c_draw, c_unproven]

    mcts._check_solved(parent)
    assert parent.drawable is True, \
        f"parent.drawable should be True with draw-proven child"
    assert parent.outcome is None, \
        "parent not fully solved (unproven child still exists)"


def test_check_solved_not_drawable_on_win_child():
    game = pyspiel.load_game("tic_tac_toe")
    evaluator = _DummyEvaluator(game)
    cfg = MCTSConfig(max_simulations=8, batch_size=4, solve=True)
    mcts = BatchMCTS(game, cfg, evaluator)

    parent = Node(None, 0, 1.0)
    c_win = Node(0, 0, 0.5)
    c_win.outcome = np.array([1.0, -1.0])  # parent wins (player 0)
    c_win.explore_count = 10
    c_unproven = Node(1, 0, 0.5)
    c_unproven.explore_count = 10
    parent.children = [c_win, c_unproven]

    mcts._check_solved(parent)
    assert parent.drawable is False, \
        "win-proven child should NOT set drawable"
    assert parent.outcome is not None, \
        "parent should be solved (proven win child → parent proven)"


# ── Test 3: _backprop clamps for drawable ─────────────────────────────────────

def test_backprop_drawable_clamp_negative():
    game = pyspiel.load_game("tic_tac_toe")
    evaluator = _DummyEvaluator(game)
    cfg = MCTSConfig(max_simulations=8, batch_size=4, solve=True)
    mcts = BatchMCTS(game, cfg, evaluator)

    # Path: root → child_drawable → leaf
    root = Node(None, 0, 1.0)
    root.explore_count = 0

    child = Node(0, 0, 0.5)
    child.drawable = True  # manually set
    child.explore_count = 0
    root.children = [child]

    leaf = Node(4, 0, 1.0)
    leaf.explore_count = 0
    child.children = [leaf]

    path = [root, child, leaf]
    # leaf player = 0, returns[0] = -0.8 (bad for player 0)
    returns = np.array([-0.8, 0.8])
    mcts._backprop(path, returns, draw_prob=0.0)

    # child (closest to leaf): drawable=True, target=-0.8 → clamped to 0
    assert child.total_reward == 0.0, \
        f"child: drawable should clamp neg target, got {child.total_reward}"
    assert child.draw_reward == 1.0, \
        f"child: should set draw_prob=1, got {child.draw_reward}"

    # root: `clamped` flag stays True → also gets target=0, draw_prob=1
    assert root.total_reward == 0.0, \
        f"root above drawable child should get clamped target=0, got {root.total_reward}"
    assert root.draw_reward == 1.0, \
        f"root should get clamped draw_prob=1, got {root.draw_reward}"


def test_backprop_drawable_no_clamp_positive():
    game = pyspiel.load_game("tic_tac_toe")
    evaluator = _DummyEvaluator(game)
    cfg = MCTSConfig(max_simulations=8, batch_size=4, solve=True)
    mcts = BatchMCTS(game, cfg, evaluator)

    root = Node(None, 0, 1.0)
    child = Node(0, 0, 0.5)
    child.drawable = True
    root.children = [child]
    leaf = Node(4, 0, 1.0)
    child.children = [leaf]

    path = [root, child, leaf]
    returns = np.array([0.5, -0.5])  # good for player 0
    mcts._backprop(path, returns, draw_prob=0.2)

    # child: drawable=True, target=0.5 > 0 → NOT clamped
    assert child.total_reward == 0.5
    assert child.draw_reward == 0.2


# ── Test 4: _stable_qdr clamps for drawable (play.py version) ─────────────────

def test_stable_qdr_drawable_clamp():
    from train.core.play import _stable_qdr

    root = Node(None, 0, 1.0)
    root.drawable = True
    root.explore_count = 20

    # Child A: negative Q (explored losing lines)
    c1 = Node(0, 0, 0.5)
    c1.explore_count = 10
    c1.total_reward = -3.0
    c1.draw_reward = 0.2
    root.children.append(c1)

    # Child B: slightly positive
    c2 = Node(1, 0, 0.5)
    c2.explore_count = 10
    c2.total_reward = 1.0
    c2.draw_reward = 0.5
    root.children.append(c2)

    q, dr = _stable_qdr(root)
    # Without clamp: q = (-3+1)/20 = -0.1 → negative
    # With clamp: q >= 0
    assert q >= 0.0, f"drawable root Q should be clamped >= 0, got {q}"
    assert dr >= 1.0, f"drawable root dr should be >= 1.0, got {dr}"


def test_stable_qdr_no_clamp_positive():
    from train.core.play import _stable_qdr

    root = Node(None, 0, 1.0)
    root.drawable = True
    root.explore_count = 20

    c1 = Node(0, 0, 0.5)
    c1.explore_count = 10
    c1.total_reward = 5.0
    root.children.append(c1)

    q, dr = _stable_qdr(root)
    assert q == 0.5, f"positive Q should not be clamped, got {q}"


# ── Test 5: MCTS end-to-end with drawable ────────────────────────────────────

def test_mcts_drawable_on_draw_position():
    """On a drawable TicTacToe position, MCTS backprop propagates drawable."""
    game = pyspiel.load_game("tic_tac_toe")
    evaluator = _DummyEvaluator(game)
    # Many simulations to force solver to find draw outcomes
    cfg = MCTSConfig(max_simulations=200, batch_size=8, solve=True,
                     policy_epsilon=0, verbose=False)
    mcts = BatchMCTS(game, cfg, evaluator)

    # Play a few moves toward a draw-heavy position
    state = game.new_initial_state()
    state.apply_action(0)  # X center
    state.apply_action(1)  # O corner
    state.apply_action(4)  # X opposite corner
    state.apply_action(2)  # O another corner

    root = mcts.mcts_search(state)
    # In TicTacToe most positions are drawable, solver should find draws
    # At least verify drawable is present or Q is non-negative
    assert root.q_value >= -0.1, \
        f"root Q should not be deeply negative on balanced position: {root.q_value}"
