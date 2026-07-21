"""Tests for Node.reparent_as_root — player flip and MCTS continuation."""

import hashlib

import numpy as np
import pyspiel
import pytest

from train.batch_mcts.node import Node
from train.batch_mcts.config import MCTSConfig
from train.batch_mcts.mcts import BatchMCTS
from train.core.surprise import _stable_qdr, _wdl_from_qdr


def _state_hash(state) -> int:
    """Deterministic hash for a pyspiel state (cross-process stable)."""
    obs = np.asarray(state.observation_tensor(), dtype=np.float32)
    digest = hashlib.md5(obs.tobytes()).digest()
    return int.from_bytes(digest[:8], "little")


class _HashEvaluator:
    """Deterministic evaluator: same state → same value/prior, every time."""

    def __init__(self, game, seed=42):
        self._game = game
        self._seed = seed
        self._prior_cache = {}

    def batch_inference_raw(self, states):
        n = len(states)
        vl = np.zeros((n, 3), dtype=np.float32)
        for i, s in enumerate(states):
            seed = (_state_hash(s) + self._seed) % (2**31)
            rng = np.random.RandomState(seed)
            w = rng.uniform(0.1, 0.5)
            d = rng.uniform(0.1, 0.4)
            vl[i] = [w, d, 1.0 - w - d]
        pl = [self.prior(s) for s in states]
        return vl, pl

    def prior(self, state):
        if state.is_chance_node():
            return state.chance_outcomes()
        h = _state_hash(state)
        if h not in self._prior_cache:
            legal = state.legal_actions(state.current_player())
            seed = (h * 31 + self._seed) % (2**31)
            rng = np.random.RandomState(seed)
            w = rng.exponential(1.0, len(legal))
            w /= w.sum()
            self._prior_cache[h] = [(a, float(w[i]))
                                    for i, a in enumerate(legal)]
        return self._prior_cache[h]


def _kl(p, q):
    """KL divergence KL(p||q) with epsilon smoothing."""
    eps = 1e-12
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def _policy_vector(policy_dict, actions):
    """Extract policy as a sorted numpy array for KL comparison."""
    return np.array([policy_dict.get(a, 0.0) for a in sorted(actions)],
                    dtype=np.float64)


# ── Unit test ───────────────────────────────────────────────────────────────

def test_reparent_as_root_unit():
    """Node fields after reparent_as_root: player flipped, reward negated,
    subtree untouched."""
    child = Node(action=3, player=0, prior=0.5)
    child.explore_count = 10
    child.total_reward = 5.0
    child.draw_reward = 2.0
    child.noise_applied = True
    child._pos_hash = 12345
    child.virtual_visits = 4

    gc = Node(action=7, player=1, prior=0.3)
    gc.explore_count = 3
    gc.total_reward = -1.5
    gc.draw_reward = 0.5
    child.children = [gc]

    child.reparent_as_root()

    assert child.action is None
    assert child.player == 1
    assert child.total_reward == -5.0
    assert child.draw_reward == 2.0   # draws symmetric, unchanged
    assert child.noise_applied is False
    assert child._pos_hash is None
    assert child.virtual_visits == 0

    # subtree untouched (same object, same stats)
    assert len(child.children) == 1
    assert child.children[0] is gc
    assert gc.player == 1
    assert gc.total_reward == -1.5
    assert gc.draw_reward == 0.5
    assert gc.explore_count == 3


# ── Integration test ────────────────────────────────────────────────────────

def test_reparent_mcts_continuation():
    """shallow search → reparent best → continue == deep search from scratch."""
    game = pyspiel.load_game("tic_tac_toe")
    rng = np.random.RandomState(42)

    cfg_deep = MCTSConfig(max_simulations=512, batch_size=16, solve=True,
                          policy_epsilon=0, uct_c=2.0, probe_depth=0)
    cfg_half = MCTSConfig(max_simulations=256, batch_size=16, solve=True,
                          policy_epsilon=0, uct_c=2.0, probe_depth=0)

    evaluator = _HashEvaluator(game, seed=42)

    # Play a couple moves for a non-trivial position
    state = game.new_initial_state()
    state.apply_action(0)
    state.apply_action(1)

    # ── Deep reference from S ──────────────────────────────────────────
    mcts_ref = BatchMCTS(game, cfg_deep, evaluator, random_state=rng)
    root_ref = mcts_ref.mcts_search(state)
    pol_ref = mcts_ref.compute_root_policy(root_ref, state)

    # ── Shallow → reparent → continue ─────────────────────────────────
    rng2 = np.random.RandomState(42)
    evaluator2 = _HashEvaluator(game, seed=42)
    mcts_s = BatchMCTS(game, cfg_half, evaluator2, random_state=rng2)
    root_s = mcts_s.mcts_search(state)

    best = root_s.best_child()
    state_next = state.clone()
    state_next.apply_action(best.action)

    best.reparent_as_root()
    explore_before = best.explore_count
    mcts_c = BatchMCTS(game, cfg_half, evaluator2, random_state=rng2)
    root_c = mcts_c.mcts_search(state_next, root=best, sim_override=256)
    pol_c = mcts_c.compute_root_policy(root_c, state_next)

    # ── Deep reference from S_next ────────────────────────────────────
    rng3 = np.random.RandomState(42)
    evaluator3 = _HashEvaluator(game, seed=42)
    mcts_d = BatchMCTS(game, cfg_deep, evaluator3, random_state=rng3)
    root_d = mcts_d.mcts_search(state_next)
    pol_d = mcts_d.compute_root_policy(root_d, state_next)

    # ── Compare policy KL ──────────────────────────────────────────────
    all_actions = sorted(set(pol_c.keys()) | set(pol_d.keys()))
    p_cont = _policy_vector(pol_c, all_actions)
    p_deep = _policy_vector(pol_d, all_actions)

    kl_pol = _kl(p_cont, p_deep)
    # Continuation is not bitwise-identical to deep search: children that
    # inherited visits from the shallow search skip the +1e6 root bonus,
    # so the first batches explore differently.  KL is still close.
    assert kl_pol < 0.4, \
        f"policy should broadly match deep reference, KL={kl_pol:.4f}"

    # ── Compare WDL KL ─────────────────────────────────────────────────
    wdl_cont = _wdl_from_qdr(*_stable_qdr(root_c))
    wdl_deep = _wdl_from_qdr(*_stable_qdr(root_d))
    kl_wdl = _kl(wdl_cont, wdl_deep)
    assert kl_wdl < 0.4, \
        f"WDL should broadly match deep reference, KL={kl_wdl:.4f}"

    # Sanity: continuation keeps or adds visits (solver may prove immediately)
    assert root_c.explore_count >= explore_before, \
        f"continuation should not lose visits (was {explore_before}, now {root_c.explore_count})"


# ── Edge cases ──────────────────────────────────────────────────────────────

def test_reparent_no_children():
    """Reparenting a leaf node should work (just flip player/reward)."""
    leaf = Node(action=5, player=0, prior=0.8)
    leaf.explore_count = 3
    leaf.total_reward = -1.2

    leaf.reparent_as_root()

    assert leaf.action is None
    assert leaf.player == 1
    assert leaf.total_reward == 1.2
    assert leaf.children == []
    assert leaf.cur_player == 1  # root without children: player == cur_player


def test_reparent_with_outcome():
    """Reparenting a proven node keeps outcome (p0/p1 indexed, not player-relative)."""
    child = Node(action=2, player=0, prior=0.6)
    child.outcome = np.array([0.0, 0.0])  # draw
    child.explore_count = 20
    child.total_reward = -3.0

    gc = Node(action=1, player=1, prior=0.5)
    gc.outcome = np.array([0.0, 0.0])
    child.children = [gc]

    child.reparent_as_root()

    assert child.player == 1
    assert child.total_reward == 3.0
    # outcome stays [0,0] — it's p0/p1 indexed, not player-relative
    assert np.array_equal(child.outcome, [0.0, 0.0])
    # cur_player check: root has children → children[0] determines it
    assert child.cur_player == 1  # gc's player


# ── MCTS integration details ─────────────────────────────────────────────────

def test_reparent_preserves_children_and_state():
    """Reparented root keeps children intact, state matches new root position."""
    game = pyspiel.load_game("tic_tac_toe")
    rng = np.random.RandomState(42)
    evaluator = _HashEvaluator(game, seed=42)
    cfg = MCTSConfig(max_simulations=64, batch_size=8, solve=True,
                     policy_epsilon=0, uct_c=2.0, probe_depth=0)

    state = game.new_initial_state()
    state.apply_action(0)
    state.apply_action(1)

    mcts = BatchMCTS(game, cfg, evaluator, random_state=rng)
    root = mcts.mcts_search(state)
    best = root.best_child()
    state_next = state.clone()
    state_next.apply_action(best.action)

    n_children = len(best.children)
    best_explore = best.explore_count

    best.reparent_as_root()

    assert len(best.children) == n_children, \
        "children should not be lost during reparent"
    assert best.explore_count == best_explore, \
        "explore_count should be unchanged"
    assert best.action is None, "should be root after reparent"

    # Children should cover all legal actions at the new root state
    legal = state_next.legal_actions()
    assert {c.action for c in best.children} == set(legal), \
        "children should cover all legal actions at the new root state"


def test_reparent_root_gets_fresh_dirichlet():
    """Reparented root gets fresh Dirichlet noise (noise_applied reset)."""
    game = pyspiel.load_game("tic_tac_toe")
    rng = np.random.RandomState(42)
    evaluator = _HashEvaluator(game, seed=42)
    cfg = MCTSConfig(max_simulations=32, batch_size=8, solve=True,
                     policy_epsilon=0.25, policy_alpha=1.0, uct_c=2.0,
                     probe_depth=0)

    state = game.new_initial_state()
    state.apply_action(0)
    state.apply_action(1)

    # Shallow search — Dirichlet already applied to this root
    mcts_s = BatchMCTS(game, cfg, evaluator, random_state=rng)
    root_s = mcts_s.mcts_search(state)
    assert root_s.noise_applied is True

    best = root_s.best_child()
    state_next = state.clone()
    state_next.apply_action(best.action)

    # Record children's priors before reparent (they're NN priors, no noise)
    priors_before = {c.action: c.prior for c in best.children}
    best.reparent_as_root()
    assert best.noise_applied is False, "reparent should reset noise flag"

    # Continue search — should apply fresh Dirichlet at the new root
    mcts_c = BatchMCTS(game, cfg, evaluator, random_state=rng)
    root_c = mcts_c.mcts_search(state_next, root=best, sim_override=64)
    assert root_c.noise_applied is True

    # At least one child's prior differs from NN prior (Dirichlet mixed in)
    nn_priors = dict(evaluator.prior(state_next))
    diffs = [abs(c.prior - nn_priors.get(c.action, 0.0))
             for c in root_c.children]
    assert max(diffs) > 0.001, \
        f"Dirichlet should alter priors, max diff={max(diffs):.4f}"


# ── Scale ────────────────────────────────────────────────────────────────────

def test_reparent_scale_preserves_q():
    """Q value stays exact after scale despite floor truncation of N."""
    child = Node(action=3, player=0, prior=0.5)
    child.explore_count = 7  # floor(7*0.5)=3 → ratio=3/7
    child.total_reward = 3.5
    child.draw_reward = 1.4

    gc = Node(action=1, player=1, prior=0.3)
    gc.explore_count = 5
    gc.total_reward = -2.0
    gc.draw_reward = 0.8
    child.children = [gc]

    q_before = abs(child.q_value)  # |3.5/7| = 0.5
    gc_q_before = gc.q_value  # -2.0/5 = -0.4 (child, no player flip)
    gc_dr_before = gc.draw_rate  # 0.8/5 = 0.16

    child.reparent_as_root(scale=0.5)

    # Q magnitude preserved despite player flip + scale truncation
    assert child.explore_count == 4  # ceil(7 * 0.5)
    assert abs(child.q_value) == pytest.approx(q_before, abs=1e-10), \
        f"|Q| should be preserved after reparent+scale: {child.q_value}"

    # Child's Q also preserved
    assert gc.explore_count == 3  # ceil(5 * 0.5)
    assert gc.q_value == pytest.approx(gc_q_before, abs=1e-10)
    assert gc.draw_rate == pytest.approx(gc_dr_before, abs=1e-10)
