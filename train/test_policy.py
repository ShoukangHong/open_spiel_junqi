"""Test solved-aware policy on manually constructed nodes."""

from train.batch_mcts.node import Node
from train.batch_mcts.mcts import compute_solved_policy


def make_child(action, explore_count=0, total_reward=0.0, outcome=None):
    n = Node(action, player=1, prior=1.0)
    n.explore_count = explore_count
    n.total_reward = total_reward
    if outcome is not None:
        n.outcome = outcome
    return n


def print_policy(case_name, children, policy, player=None):
    print(f"\n  -- {case_name} --")
    if player is not None:
        print(f"  (player={player})")
    print(f"  {'action':>6s}  {'explore':>7s}  {'Q':>8s}  "
          f"{'outcome':>12s}  {'policy':>8s}")
    for c in children:
        p = policy.get(c.action, 0.0)
        o_str = (f"({c.outcome[0]:+.0f},{c.outcome[1]:+.0f})"
                 if c.outcome is not None else "-")
        print(f"  {c.action:>6d}  {c.explore_count:>7d}  {c.q_value:+8.4f}  "
              f"{o_str:>12s}  {p:8.4f}")


PLAYER = 1        # O
MAX_UTIL = 1.0
OUT_WIN  = (-1.0, 1.0)
OUT_DRAW = (0.0, 0.0)
OUT_LOSS = (1.0, -1.0)


def run_tests():
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("Case 1: Fully solved — best outcome splits evenly")
    print("=" * 60)

    c1a = [
        make_child(2, 100,  0.0,  OUT_DRAW),
        make_child(6, 80,   0.0,  OUT_DRAW),
        make_child(1, 50,  -50.0, OUT_LOSS),
        make_child(7, 40,  -40.0, OUT_LOSS),
    ]
    p = compute_solved_policy(c1a, PLAYER, MAX_UTIL)
    print_policy("O: 1a: 2 DRAW + 2 LOSS", c1a, p)
    assert abs(p[2] - 0.5) < 1e-9 and abs(p[6] - 0.5) < 1e-9
    assert p[1] == 0.0 and p[7] == 0.0
    print("  PASSED")

    c1b = [
        make_child(2, 100, 100.0, OUT_WIN),
        make_child(6, 80,    0.0, OUT_DRAW),
        make_child(7, 40,  -40.0, OUT_LOSS),
    ]
    p = compute_solved_policy(c1b, PLAYER, MAX_UTIL)
    print_policy("O: 1b: 1 WIN + 1 DRAW + 1 LOSS", c1b, p)
    assert abs(p[2] - 1.0) < 1e-9
    assert p[6] == 0.0 and p[7] == 0.0
    print("  PASSED")

    c1c = [
        make_child(1, 50, -50.0, OUT_LOSS),
        make_child(3, 40, -40.0, OUT_LOSS),
    ]
    p = compute_solved_policy(c1c, PLAYER, MAX_UTIL)
    print_policy("O: 1c: all LOSS", c1c, p)
    assert abs(p[1] - 0.5) < 1e-9 and abs(p[3] - 0.5) < 1e-9
    print("  PASSED")

    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Case 2: Reward/Penalty (non-loss proven exists)")
    print("=" * 60)

    # 2a: Proven DRAW + proven LOSS(high conf) + unvisited
    #   DRAW(eff=0,wt=1), LOSS(eff=-1,conf=14.14,wt=exp(-28)≈0), unvisited(eff=0,wt=1)
    c2a = [
        make_child(2, 200, -10.0,  OUT_DRAW),
        make_child(5, 200, -200.0, OUT_LOSS),
        make_child(7, 0,    0.0,   None),
    ]
    p = compute_solved_policy(c2a, PLAYER, MAX_UTIL)
    print_policy("O: 2a: DRAW vs LOSS(200v) vs unvisited", c2a, p)
    assert abs(p[2] - 0.5) < 0.01
    assert p[5] < 1e-6
    print("  PASSED")

    # 2b: Proven LOSS(few visits, light penalty)
    #   LOSS(5v): eff=-1, diff=-1, conf=2.24, wt=exp(-4.47)=0.011
    c2b = [
        make_child(2, 100, -5.0,  OUT_DRAW),
        make_child(5, 5,   -5.0,  OUT_LOSS),
        make_child(7, 0,    0.0,  None),
    ]
    p = compute_solved_policy(c2b, PLAYER, MAX_UTIL)
    print_policy("O: 2b: DRAW vs LOSS(5v) vs unvisited", c2b, p)
    assert 0.005 < p[5] < 0.05, f"LOSS(5v) policy={p[5]:.4f}"
    assert p[2] > p[5]
    print("  PASSED")

    # 2c: Proven DRAW vs unproven(slightly worse Q)
    #   unproven: eff=-0.06, diff=-0.06, conf=7.07, wt=exp(-0.85)=0.427
    c2c = [
        make_child(2, 50, -2.5,  OUT_DRAW),
        make_child(6, 50, -3.0,  None),
        make_child(5, 50, -50.0, OUT_LOSS),
    ]
    p = compute_solved_policy(c2c, PLAYER, MAX_UTIL)
    print_policy("O: 2c: DRAW vs unproven(Q=-0.06) vs LOSS", c2c, p)
    assert p[2] > 0.65, f"DRAW={p[2]:.3f}"
    assert 0.2 < p[6] < 0.35, f"unproven={p[6]:.3f}"
    assert p[5] < 0.01
    print("  PASSED")

    # 2d: Proven DRAW(few visits) vs unproven(good Q, many visits)
    #   DRAW: eff=0,wt=1. Unproven: eff=-0.02, diff=-0.02, conf=14.14, wt=exp(-0.57)=0.568
    c2d = [
        make_child(2, 10,   0.0, OUT_DRAW),
        make_child(6, 200, -4.0, None),
    ]
    p = compute_solved_policy(c2d, PLAYER, MAX_UTIL)
    print_policy("O: 2d: DRAW(10v) vs unproven(Q=-0.02, 200v)", c2d, p)
    assert p[2] > 0.6, f"DRAW={p[2]:.3f}"
    assert p[6] > 0.3, f"unproven={p[6]:.3f}"
    print("  PASSED")

    # 2e: Proven WIN dominates
    #   WIN(eff=1): diff=+1, conf=3.16, wt=exp(6.32)=556 → >99.9%
    c2e = [
        make_child(2, 10,  10.0,  OUT_WIN),
        make_child(6, 200, -4.0,  None),
        make_child(5, 200, -200.0, OUT_LOSS),
    ]
    p = compute_solved_policy(c2e, PLAYER, MAX_UTIL)
    print_policy("O: 2e: WIN vs unproven vs LOSS", c2e, p)
    assert p[2] > 0.999, f"WIN={p[2]:.4f}"
    print("  PASSED")

    # 2f: LOSS(high-conf) crushed harder than LOSS(low-conf)
    #   LOSS(200v): diff=-1,conf=14.14,wt=exp(-28.3)≈0
    #   LOSS(10v): diff=-1,conf=3.16,wt=exp(-6.32)=0.0018
    c2f = [
        make_child(2, 200, -10.0, OUT_DRAW),
        make_child(6, 50,  -2.0,  OUT_DRAW),
        make_child(1, 200, -200.0, OUT_LOSS),
        make_child(3, 10,  -10.0,  OUT_LOSS),
    ]
    p = compute_solved_policy(c2f, PLAYER, MAX_UTIL)
    print_policy("O: 2f: 2 DRAW + LOSS(200v) + LOSS(10v)", c2f, p)
    assert p[2] + p[6] > 0.99, f"DRAW total={p[2]+p[6]:.4f}"
    # Both LOSS have tiny weight; 10v (conf=3.16) > 200v (conf=14.14)
    assert p[3] < 0.001 and p[1] < 1e-6, \
        f"LOSS: 10v={p[3]:.6f} 200v={p[1]:.6f}"
    print("  PASSED")

    # 2g: Unproven with BETTER Q than proven DRAW → gets reward
    #   unproven(Q=+0.05, 100v): eff=+0.05, diff=+0.05, conf=10, wt=exp(1.0)=2.718
    #   DRAW(eff=0): wt=1.  unproven=73%
    c2g = [
        make_child(2, 50,  -2.5, OUT_DRAW),
        make_child(6, 100, +5.0, None),
    ]
    p = compute_solved_policy(c2g, PLAYER, MAX_UTIL)
    print_policy("O: 2g: DRAW(eff=0) vs unproven(eff=+0.05, 100v)", c2g, p)
    assert p[6] > p[2], f"unproven={p[6]:.3f} should lead DRAW={p[2]:.3f}"
    assert p[6] > 0.6, f"unproven should dominate, got {p[6]:.3f}"
    print("  PASSED")

    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Case 3: No non-loss proven (visit-prop, LOSS zeroed)")
    print("=" * 60)

    c3a = [
        make_child(2, 100, -5.0,  None),
        make_child(6, 50,  -2.0,  None),
        make_child(5, 80,  -80.0, OUT_LOSS),
        make_child(7, 60,  -60.0, OUT_LOSS),
    ]
    p = compute_solved_policy(c3a, PLAYER, MAX_UTIL)
    print_policy("O: 3a: unproven + proven LOSS", c3a, p)
    assert p[5] == 0.0 and p[7] == 0.0
    assert abs(p[2] - 100/150) < 0.01
    assert abs(p[6] - 50/150) < 0.01
    print("  PASSED")

    c3b = [
        make_child(2, 200, -10.0, None),
        make_child(6, 100, -5.0,  None),
        make_child(5, 100, -50.0, None),
    ]
    p = compute_solved_policy(c3b, PLAYER, MAX_UTIL)
    print_policy("O: 3b: all unproven (visit-prop)", c3b, p)
    assert abs(p[2] - 0.5) < 0.01
    assert abs(p[6] - 0.25) < 0.01
    assert abs(p[5] - 0.25) < 0.01
    print("  PASSED")

    c3c = [
        make_child(1, 100, -100.0, OUT_LOSS),
        make_child(3, 50,  -50.0,  OUT_LOSS),
        make_child(5, 150, -150.0, OUT_LOSS),
    ]
    p = compute_solved_policy(c3c, PLAYER, MAX_UTIL)
    print_policy("O: 3c: all proven LOSS (uniform)", c3c, p)
    for a in [1, 3, 5]:
        assert abs(p[a] - 1/3) < 0.01
    print("  PASSED")

    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Both-player symmetry: verify player 0 (X) perspective")
    print("=" * 60)

    # X-player children: player=0. WIN=OUT_X_WIN, DRAW=OUT_DRAW, LOSS=OUT_X_LOSS.
    OUT_X_WIN  = (1.0, -1.0)
    OUT_X_LOSS = (-1.0, 1.0)

    def make_x(action, explore_count=0, total_reward=0.0, outcome=None):
        n = Node(action, player=0, prior=1.0)
        n.explore_count = explore_count
        n.total_reward = total_reward
        if outcome is not None:
            n.outcome = outcome
        return n

    # Case 1 for X: 2 WIN + 1 LOSS
    cx1 = [
        make_x(0, 100, 100.0,  OUT_X_WIN),
        make_x(4, 80,  80.0,  OUT_X_WIN),
        make_x(8, 40, -40.0,  OUT_X_LOSS),
    ]
    p = compute_solved_policy(cx1, 0, MAX_UTIL)
    print_policy("X: 2 WIN + 1 LOSS", cx1, p, player=0)
    assert abs(p[0] - 0.5) < 1e-9
    assert abs(p[4] - 0.5) < 1e-9
    assert p[8] == 0.0
    print("  PASSED")

    # Case 2 for X: WIN + unproven(good Q) + LOSS
    cx2 = [
        make_x(0, 10,  10.0,  OUT_X_WIN),
        make_x(4, 200, -4.0,  None),
        make_x(8, 200, -200.0, OUT_X_LOSS),
    ]
    p = compute_solved_policy(cx2, 0, MAX_UTIL)
    print_policy("X: WIN vs unproven(Q=-0.02) vs LOSS", cx2, p, player=0)
    assert p[0] > 0.999, f"X WIN should dominate, got {p[0]:.4f}"
    print("  PASSED")

    # Case 3 for X: unproven + LOSS — LOSS zeroed
    cx3 = [
        make_x(0, 200, -10.0, None),
        make_x(4, 100, -5.0,  None),
        make_x(8, 80,  -80.0, OUT_X_LOSS),
    ]
    p = compute_solved_policy(cx3, 0, MAX_UTIL)
    print_policy("X: unproven + LOSS", cx3, p, player=0)
    assert p[8] == 0.0
    assert abs(p[0] - 200/300) < 0.01
    assert abs(p[4] - 100/300) < 0.01
    print("  PASSED")

    print(f"\n{'=' * 60}")
    print("ALL TESTS PASSED")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_tests()
