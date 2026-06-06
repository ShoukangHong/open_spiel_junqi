"""Elo rating analysis from training eval logs.

Parses [eval] lines from train.log, extracts pairwise W/D/L records,
then uses bootstrap simulation to compute stable Elo ratings for each
checkpoint step.

Usage:
    python experiment/elo_analysis.py                          # default paths
    python experiment/elo_analysis.py --log path/to/train.log  # custom log
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# 1. Parse eval lines
# ---------------------------------------------------------------------------

_EVAL_RE = re.compile(
    r"\[eval\]\s+step(\d+)\s+vs\s+(step(\d+)|random):\s+"
    r"W=(\d+)\s+L=(\d+)\s+D=(\d+)\s+WR=([\d.]+)%"
)


def parse_eval_log(log_path: str) -> dict:
    """Parse train.log, return {model_name: {opponent_name: (W, D, L)}}."""
    records = defaultdict(lambda: defaultdict(lambda: (0, 0, 0)))

    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _EVAL_RE.search(line)
            if not m:
                continue
            step_a = int(m.group(1))
            opp_raw = m.group(2)
            step_b = int(m.group(3)) if m.group(3) else "random"
            w, l, d = int(m.group(4)), int(m.group(5)), int(m.group(6))

            name_a = f"step{step_a}"
            name_b = f"step{step_b}" if isinstance(step_b, int) else "random"

            records[name_a][name_b] = (w, d, l)
            records[name_b][name_a] = (l, d, w)  # reverse perspective

    return dict(records)


# ---------------------------------------------------------------------------
# 2. Build pairwise WDL dict for bootstrap
# ---------------------------------------------------------------------------

def build_matchup_dict(records: dict) -> dict:
    """Convert records into { (a, b): (p_win_a, p_draw, p_win_b) }.

    Only includes matchups where the total games > 0.
    """
    matchups = {}
    seen = set()
    for a, opponents in records.items():
        for b, (w, d, l) in opponents.items():
            pair = tuple(sorted([a, b]))
            if pair in seen:
                continue
            seen.add(pair)
            total = w + d + l
            if total == 0:
                continue
            # Store WDL from the perspective of pair[0]
            if a == pair[0]:
                matchups[pair] = (w / total, d / total, l / total)
            else:
                matchups[pair] = (l / total, d / total, w / total)
    return matchups


# ---------------------------------------------------------------------------
# 3. Elo simulation
# ---------------------------------------------------------------------------

def expected_score(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def simulate_elo(matchups: dict, n_rounds: int = 5000, K: float = 16.0,
                 games_per_pair: int = 100, seed: int = 42) -> dict:
    """Bootstrap Elo ratings by repeatedly simulating tournaments.

    Each round simulates a full round-robin: every pair plays *games_per_pair*
    games, and Elo is updated from the aggregate result.  This keeps variance
    low while converging to the correct equilibrium.

    Args:
        matchups: {(a, b): (p_win_a, p_draw, p_win_b)}  from pair[0]'s view.
        n_rounds: number of tournament cycles.
        K: Elo K-factor.
        games_per_pair: number of simulated games per pair per round.

    Returns:
        {model_name: [elo_after_each_round]} — trajectory for each model.
    """
    rng = np.random.default_rng(seed)

    # Collect all model names
    models_set = set()
    for a, b in matchups:
        models_set.add(a)
        models_set.add(b)
    models = sorted(models_set, key=_model_sort_key)

    elo = {m: 1000.0 for m in models}
    trajectories = {m: [1000.0] for m in models}

    pairs = list(matchups.keys())
    probs = list(matchups.values())

    # Cumulative tournament results for diagnostic logging
    cum_wins = {p: 0 for p in pairs}
    cum_games = {p: 0 for p in pairs}

    for round_idx in range(n_rounds):
        # Shuffle pair order each round to avoid order bias
        order = rng.permutation(len(pairs))
        for idx in order:
            a, b = pairs[idx]
            p_w, p_d, p_l = probs[idx]

            # Simulate many games and aggregate
            samples = rng.random(games_per_pair)
            wins_a = int((samples < p_w).sum())
            draws = int(((samples >= p_w) & (samples < p_w + p_d)).sum())
            # losses_a = int((samples >= p_w + p_d).sum())  # = rest

            cum_wins[(a, b)] += wins_a
            cum_games[(a, b)] += games_per_pair

            score_a = (wins_a + 0.5 * draws) / games_per_pair
            score_b = 1.0 - score_a

            ea = expected_score(elo[a], elo[b])
            elo[a] += K * (score_a - ea)
            elo[b] += K * (score_b - (1.0 - ea))

        for m in models:
            trajectories[m].append(elo[m])

    return trajectories


def _model_sort_key(name: str):
    if name == "random":
        return -1
    return int(name.replace("step", ""))


# ---------------------------------------------------------------------------
# 4. Stability analysis
# ---------------------------------------------------------------------------

def check_stability(trajectories: dict, window: int = 200,
                    threshold: float = 2.0) -> bool:
    """Check if Elo ratings have stabilised in the last `window` rounds."""
    for name, traj in trajectories.items():
        if len(traj) < window:
            return False
        recent = traj[-window:]
        if np.ptp(recent) > threshold:
            return False
    return True


# ---------------------------------------------------------------------------
# 5. Plotting
# ---------------------------------------------------------------------------

def plot_elo(trajectories: dict, n_rounds: int, output_path: str = None):
    """Plot Elo trajectories (bootstrap rounds) and final Elo vs step."""
    models = sorted(trajectories.keys(), key=_model_sort_key)
    final_elo = {m: trajectories[m][-1] for m in models}

    fig, axes = plt.subplots(2, 1, figsize=(7, 10))

    # --- Left: Elo evolution over bootstrap rounds ---
    ax = axes[0]
    for m in models:
        ax.plot(trajectories[m], alpha=0.7, linewidth=0.8,
                label=m if len(models) <= 20 else None)
    ax.set_xlabel("bootstrap round")
    ax.set_ylabel("Elo")
    ax.set_title(f"Elo trajectories ({n_rounds} rounds)")
    ax.grid(True, alpha=0.3)
    if len(models) <= 20:
        ax.legend(fontsize=7, ncol=2)

    # --- Right: Final Elo vs step ---
    ax = axes[1]
    step_nums = []
    elo_vals = []
    colors = []
    for m in models:
        step_nums.append(_model_sort_key(m))
        elo_vals.append(final_elo[m])
        colors.append("#999999" if m == "random" else "#2196F3")
    ax.bar(range(len(models)), elo_vals, color=colors, width=0.7)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m if i % max(1, len(models) // 20) == 0 else ""
                         for i, m in enumerate(models)],
                       rotation=45, fontsize=7)
    ax.set_ylabel("Elo")
    ax.set_title("Final Elo by checkpoint")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1000, color="gray", linestyle="--", alpha=0.5)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[elo] Plot saved to {output_path}")
    plt.show()


# ---------------------------------------------------------------------------
# 6. Report
# ---------------------------------------------------------------------------

def print_report(records: dict, trajectories: dict):
    """Print match records and final Elo table."""
    models = sorted(trajectories.keys(), key=_model_sort_key)
    final_elo = {m: trajectories[m][-1] for m in models}

    print("\n=== Pairwise match records ===")
    seen = set()
    for a in sorted(records.keys(), key=_model_sort_key):
        for b in sorted(records[a].keys(), key=_model_sort_key):
            pair = tuple(sorted([a, b]))
            if pair in seen:
                continue
            seen.add(pair)
            w, d, l = records[a][b]
            total = w + d + l
            if total == 0:
                continue
            wr = w / total * 100
            print(f"  {a:>10s} vs {b:<10s}  "
                  f"W={w:3d} L={l:3d} D={d:3d}  "
                  f"WR={wr:5.1f}%  ({total} games)")

    print("\n=== Elo ratings ===")
    print(f"  {'Model':>10s}  {'Elo':>8s}  {'Δ from 1000':>12s}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*12}")
    for m in models:
        delta = final_elo[m] - 1000
        print(f"  {m:>10s}  {final_elo[m]:8.1f}  {delta:+11.1f}")

# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Elo analysis from AlphaZero eval logs")
    parser.add_argument("--log", type=str, default=DEFAULT_LOG,
                        help="Path to train.log (auto-detected if omitted)")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS,
                        help="Tournament cycles (default: 500)")
    parser.add_argument("--games-per-pair", type=int, default=DEFAULT_GAMES_PER_PAIR,
                        help="Simulated games per pair per round (default: 200)")
    parser.add_argument("--K", type=float, default=DEFAULT_K,
                        help="Elo K-factor (default: 32)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save the plot (default: <log>_elo.png)")
    args = parser.parse_args()

    # Auto-detect log path
    log_path = args.log
    if log_path is None:
        print("[elo] No log file found. Specify --log PATH.")
        sys.exit(1)

    # Parse
    records = parse_eval_log(log_path)
    if not records:
        print("[elo] No eval lines found in log.")
        sys.exit(1)

    print(f"[elo] Found {len(records)} models with eval records")

    # Build WDL pairwise dict
    matchups = build_matchup_dict(records)
    print(f"[elo] {len(matchups)} unique matchups")

    # Simulate
    trajectories = simulate_elo(matchups, n_rounds=args.rounds, K=args.K,
                                games_per_pair=args.games_per_pair)
    stable = check_stability(trajectories)
    print(f"[elo] Ran {args.rounds} tournament cycles "
          f"({args.rounds * args.games_per_pair} effective games per pair), "
          f"stable={'yes' if stable else 'no'}")

    # Report
    print_report(records, trajectories)

    # Plot
    output_path = args.output or str(Path(log_path).with_suffix("")) + "_elo.png"
    plot_elo(trajectories, args.rounds, output_path)


DEFAULT_LOG = r"C:\Users\shouk\othello_train\cloud_wdl_128\train.log"
DEFAULT_ROUNDS = 5000
DEFAULT_GAMES_PER_PAIR = 100
DEFAULT_K = 16

if __name__ == "__main__":
    main()
