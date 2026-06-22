"""Measure MCTS policy reproducibility.

Runs 500-visit MCTS twice on each of 100 positions (with Dirichlet noise),
computing the KL divergence between the two visit distributions.  A high
KL (>0.5) means the MCTS search target itself is unstable — the model
can't possibly fit a noisy signal (noise floor).  A low KL (<0.1) means
the search is stable and the model should be able to learn.

Usage:
    python -m experiment.mcts_reproducibility
    python -m experiment.mcts_reproducibility --positions 200
"""

import argparse
import numpy as np
import pyspiel

from game_ui.core import load_model_for_ui, create_bot
from train.core.model_builder import build_xiangqi_model
from train.batch_mcts.mcts import BatchMCTS

# ── Config (mirrors xiangqi_ui) ──────────────────────────────────────────────

CHECKPOINT_DIR = r"C:\Users\shouk\xiangqi_train\cloud_fpu"
CHECKPOINT_STEP = 60
MCTS_SIMS = 500
BATCH_SIZE = 10
UCT_C = 4.0
FPU_LAMBDA = 0.2
POLICY_EPSILON = 0.25
POLICY_ALPHA = 0.25
DRAW_PENALTY = 0.2
REPEAT_PENALTY = 0.1
NUM_POSITIONS = 100


def _distribution_kl(p, q):
    """Symmetric KL between two prob distributions (numpy arrays)."""
    eps = 1e-12
    kl_pq = np.sum(p * np.log((p + eps) / (q + eps)))
    kl_qp = np.sum(q * np.log((q + eps) / (p + eps)))
    return 0.5 * (kl_pq + kl_qp)


def _roots_to_dist(root1, root2):
    """Extract aligned prob distributions from two MCTS roots."""
    all_actions = sorted({c.action for c in root1.children}
                         | {c.action for c in root2.children})
    n1 = {c.action: c.explore_count for c in root1.children}
    n2 = {c.action: c.explore_count for c in root2.children}
    p = np.array([n1.get(a, 0) for a in all_actions], dtype=np.float64)
    q = np.array([n2.get(a, 0) for a in all_actions], dtype=np.float64)
    p /= p.sum()
    q /= q.sum()
    return p, q, all_actions


def _nn_dist(root, state, evaluator):
    """Get NN prior distribution aligned to root actions."""
    all_actions = sorted({c.action for c in root.children})
    n = {c.action: c.prior for c in root.children}
    p = np.array([max(n.get(a, 0), 0.0) for a in all_actions], dtype=np.float64)
    p /= p.sum()
    return p, all_actions


def generate_positions(game, model, evaluator, n):
    """Generate *n* mid-game positions via NN-prior self-play."""
    positions = []
    rng = np.random.RandomState(42)

    while len(positions) < n:
        state = game.new_initial_state()
        for move_count in range(80):
            if state.is_terminal():
                break
            obs = np.asarray(state.observation_tensor(), dtype=np.float32)
            mask = np.asarray(state.legal_actions_mask(), dtype=bool)
            _, policy = model.inference(obs, mask)
            # Sample from NN prior (τ=1.0 for diversity)
            leg = state.legal_actions()
            probs = np.array([policy[a] for a in leg], dtype=np.float64)
            probs /= probs.sum()
            a = leg[rng.choice(len(leg), p=probs)]
            state.apply_action(a)
            if 10 <= move_count <= 30 and not state.is_terminal():
                positions.append(state.clone())
        if len(positions) >= n:
            break
    return positions[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=int, default=NUM_POSITIONS)
    args = parser.parse_args()

    print(f"Loading model checkpoint-{CHECKPOINT_STEP} ...")
    model, game = load_model_for_ui(CHECKPOINT_DIR, CHECKPOINT_STEP,
                                    build_xiangqi_model)
    print(f"  params={model.num_trainable_variables}")

    # Generate positions
    bot, evaluator, cfg = create_bot(
        game, model, MCTS_SIMS, BATCH_SIZE, UCT_C,
        draw_penalty=DRAW_PENALTY, repeat_penalty=REPEAT_PENALTY,
        policy_epsilon=POLICY_EPSILON, policy_alpha=POLICY_ALPHA,
        fpu_lambda=FPU_LAMBDA)

    print(f"Generating {args.positions} positions via self-play ...")
    positions = generate_positions(game, model, evaluator, args.positions)
    print(f"  Generated {len(positions)} positions")

    # Run paired MCTS + NN comparison
    m2m_kls = []
    n2m_kls = []
    print(f"Running {len(positions)} × 3 MCTS searches ...")
    for i, state in enumerate(positions):
        mcts1 = BatchMCTS(game, cfg, evaluator,
                          random_state=np.random.RandomState(1000 + i))
        mcts2 = BatchMCTS(game, cfg, evaluator,
                          random_state=np.random.RandomState(2000 + i))
        r1 = mcts1.mcts_search(state.clone())
        r2 = mcts2.mcts_search(state.clone())
        # MCTS-MCTS KL
        pm, qm, _ = _roots_to_dist(r1, r2)
        m2m_kls.append(_distribution_kl(pm, qm))
        # NN-MCTS KL (average over both MCTS runs)
        pn, _ = _nn_dist(r1, state, evaluator)
        n2m1 = _distribution_kl(pn, pm)
        n2m2 = _distribution_kl(pn, qm)
        n2m_kls.append(0.5 * (n2m1 + n2m2))
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(positions)}  "
                  f"M2M_KL={np.mean(m2m_kls):.4f}  "
                  f"N2M_KL={np.mean(n2m_kls):.4f}")

    m2m = np.array(m2m_kls)
    n2m = np.array(n2m_kls)
    print(f"\n{'='*50}")
    print(f"Results ({len(m2m)} positions, {MCTS_SIMS} sims each):")
    print(f"\n  MCTS-MCTS KL (search reproducibility):")
    print(f"    mean={m2m.mean():.4f}  median={np.median(m2m):.4f}  "
          f"std={m2m.std():.4f}")
    print(f"    <0.1:{(m2m<0.1).mean():.0%}  <0.2:{(m2m<0.2).mean():.0%}  "
          f">0.5:{(m2m>0.5).mean():.0%}")
    print(f"\n  NN-MCTS KL (model vs search divergence):")
    print(f"    mean={n2m.mean():.4f}  median={np.median(n2m):.4f}  "
          f"std={n2m.std():.4f}")
    print(f"    <0.1:{(n2m<0.1).mean():.0%}  <0.2:{(n2m<0.2).mean():.0%}  "
          f">0.5:{(n2m>0.5).mean():.0%}")
    print(f"\nInterpretation:")
    print(f"  M2M ≪ N2M → model can't fit MCTS (model problem)")
    print(f"  M2M ≈ N2M → MCTS IS the noise floor (data problem)")


if __name__ == "__main__":
    main()
