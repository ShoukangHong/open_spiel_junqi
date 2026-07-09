"""Analyze KL(one_hot || nn_prior) for one-move-checkmate scenarios.

When MCTS policy collapses to one-hot (all visits on the winning move),
KL(MCTS || NN) = -log(NN_prior_of_winning_move).

This experiment maps out what KL values to expect across different:
  - NN prior values for the winning move
  - Number of legal actions (with uniform prior on non-winning moves)
  - Common surprise thresholds
"""

import numpy as np

# Reproduce surprise.py's _kl exactly
def _kl(p, q):
    """KL divergence KL(p||q) — matches train/core/surprise.py."""
    eps = 1e-12
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def one_hot_kl(nn_prior_of_best, num_actions=40, uniform_rest=True):
    """KL(one_hot || nn_prior) for a one-hot MCTS policy.

    Args:
        nn_prior_of_best: NN's prior probability for the winning action
        num_actions: total number of legal actions
        uniform_rest: if True, remaining probability is spread uniformly;
                      if False, all remaining mass is on one runner-up

    Returns:
        KL value and the full NN prior distribution
    """
    if uniform_rest:
        nn_prior = np.full(num_actions,
                           (1.0 - nn_prior_of_best) / (num_actions - 1),
                           dtype=np.float64)
    else:
        nn_prior = np.zeros(num_actions, dtype=np.float64)
        if num_actions > 1:
            nn_prior[1] = 1.0 - nn_prior_of_best
    nn_prior[0] = nn_prior_of_best

    mcts = np.zeros(num_actions, dtype=np.float64)
    mcts[0] = 1.0

    return _kl(mcts, nn_prior), nn_prior


def main():
    # ── Common surprise thresholds ──────────────────────────────────
    thresholds = [0.1, 0.3, 0.5, 1.0]
    print("=" * 72)
    print("One-Hot MCTS Policy → KL Analysis")
    print("KL(MCTS_one_hot || NN_prior) = -log(NN_prior_of_winning_move)")
    print("=" * 72)

    # ── 1. KL vs NN prior for the winning move (analytical) ────────
    print("\n── 1. KL vs NN prior for the winning action ───────────────")
    print(f"    (independent of num_actions, uniform rest)")
    print(f"{'NN_prior(best)':>16s}  {'KL':>8s}  {'trigger surprise?':>25s}")
    print("-" * 56)
    for p_best in [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
                   0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
        kl, _ = one_hot_kl(p_best, num_actions=40)
        triggers = []
        for t in thresholds:
            triggers.append(f"{'!!' if kl > t else 'ok'}(>{t})")
        trigger_str = ", ".join(triggers)
        print(f"{p_best:>16.4f}  {kl:>8.4f}  {trigger_str}")

    # ── 2. Analytical formula ──────────────────────────────────────
    print("\n── 2. Analytical: KL = -log(p_best) ────────────────────────")
    for p_best in [0.99, 0.9, 0.5, 0.1, 0.01, 0.001]:
        actual, _ = one_hot_kl(p_best, num_actions=40)
        predicted = -np.log(p_best)
        print(f"  p_best={p_best:<8.4f}  KL={actual:.6f}  -log(p)={predicted:.6f}  match={np.isclose(actual, predicted)}")

    # ── 3. KL vs num_actions with uniform NN prior ──────────────────
    print("\n── 3. KL vs num_actions (uniform NN prior) ─────────────────")
    print(f"    KL = log(num_actions)")
    print(f"{'num_actions':>14s}  {'KL':>8s}  {'note':>20s}")
    print("-" * 50)
    for n in [2, 5, 10, 20, 30, 40, 50, 60, 80, 100, 200, 500, 1000, 8100]:
        kl, _ = one_hot_kl(1.0 / n, num_actions=n)
        triggers = []
        for t in thresholds:
            triggers.append(f"{'SURPRISE' if kl > t else 'ok'}")
        print(f"{n:>14d}  {kl:>8.4f}  {', '.join(triggers)}")

    # ── 4. How high must NN prior be to avoid surprise? ────────────
    print("\n── 4. Min NN prior of best action to stay BELOW threshold ──")
    print(f"    (p_min = exp(-threshold))")
    print(f"{'threshold':>12s}  {'p_min_needed':>14s}")
    print("-" * 32)
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0]:
        p_min = np.exp(-t)
        print(f"{t:>12.4f}  {p_min:>14.4f}")

    # ── 5. Sensitivity: how KL grows as NN prior drops ─────────────
    print("\n── 5. KL sensitivity around threshold region ───────────────")
    print(f"    Xiangqi typical thresholds: pol_kl=0.5, val_kl=0.5")
    print(f"    p_min to avoid surprise: exp(-0.5) = {np.exp(-0.5):.4f}")
    print(f"{'delta':>8s}  {'p_best':>10s}  {'KL':>8s}  {'verdict':>20s}")
    print("-" * 56)
    p_ref = np.exp(-0.5)  # ~0.6065
    for delta in [-0.2, -0.1, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.1, 0.2]:
        p_best = np.clip(p_ref + delta, 0.001, 0.999)
        kl, _ = one_hot_kl(p_best, num_actions=40)
        verdict = "surprise!" if kl > 0.5 else "ok"
        print(f"{delta:>+8.4f}  {p_best:>10.4f}  {kl:>8.4f}  {verdict:>20s}")

    # ── 6. Non-uniform NN prior: runner-up vs uniform rest ──────────
    print("\n── 6. Uniform rest vs one runner-up (num_actions=40) ───────")
    print(f"{'p_best':>10s}  {'KL(uniform)':>12s}  {'KL(1-runner)':>12s}  {'diff':>8s}")
    print("-" * 52)
    for p_best in [0.9, 0.7, 0.5, 0.3, 0.1, 0.05]:
        kl_u, _ = one_hot_kl(p_best, num_actions=40, uniform_rest=True)
        kl_r, _ = one_hot_kl(p_best, num_actions=40, uniform_rest=False)
        diff = kl_r - kl_u
        print(f"{p_best:>10.4f}  {kl_u:>12.4f}  {kl_r:>12.4f}  {diff:>+8.4f}")

    # ── 7. Summary: what this means for one-move-checkmate ─────────
    print("\n── 7. Practical takeaway ────────────────────────────────────")
    print("""
    For a one-move-checkmate position:
    - If NN gives the winning move prior >= 0.61 → KL ≤ 0.5 (no surprise)
    - If NN gives the winning move prior = 0.40 → KL = 0.92 (surprise)
    - If NN gives the winning move prior = 0.10 → KL = 2.30 (super_surprise)
    - Uniform over 40 actions (prior=0.025) → KL = 3.69

    To avoid flagging these as surprise, the NN just needs to rank the
    mating move as its #1 choice with a reasonable prior (~0.6+).
    It does NOT need a one-hot prior to pass the KL threshold.
    """)

    # ── 8. How often is the mating move #1 vs #k in NN prior? ──────
    print("── 8. KL if winning move is ranked #k in NN prior ──────────")
    print("    (NN prior is uniform over top-K, rest uniform)")
    print(f"{'rank':>6s}  {'p_best':>10s}  {'KL':>8s}  {'verdict(0.5)':>15s}")
    print("-" * 50)
    for rank_k in [1, 2, 3, 5, 10, 20]:
        # NN prior: top-K actions share 80% of mass, winning move gets 1/K of that
        top_mass = 0.8
        p_best = top_mass / rank_k
        # Construct distribution
        nn = np.ones(40, dtype=np.float64) * (0.2 / 39)  # rest uniform
        nn[:rank_k] = top_mass / rank_k  # top K uniform
        mcts = np.zeros(40, dtype=np.float64)
        mcts[0] = 1.0
        kl = _kl(mcts, nn)
        verdict = "SURPRISE" if kl > 0.5 else "ok"
        print(f"{rank_k:>6d}  {p_best:>10.4f}  {kl:>8.4f}  {verdict:>15s}")


if __name__ == "__main__":
    main()
