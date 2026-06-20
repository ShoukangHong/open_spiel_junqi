"""Training metrics dashboard from AlphaZero train.log.

Parses per-step training metrics and produces:
  - Loss curves (policy, value, total)
  - Entropy
  - Throughput (states/s)
  - Game outcomes (p0 win%, p1 win%, draw%)
  - Weak-move statistics
  - Buffer health

Usage:
    python experiment/training_metrics.py                          # auto-detect log
    python experiment/training_metrics.py --log path/to/train.log  # custom log
    python experiment/training_metrics.py --smooth 3               # moving-average window
"""

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, fields
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# 1. Data structures
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    step: int
    games: int
    states: int
    buffer_current: int
    buffer_total: int
    unique_states: int
    tags_normal: int = 0
    tags_rare: int = 0
    rare_count: int = 0
    rf_count: int = 0
    wf_count: int = 0
    weak_count: int = 0
    rare_per_game: float = 0.0
    rf_per_game: float = 0.0
    wf_per_game: float = 0.0
    weak_per_game: float = 0.0
    states_per_s: float = 0.0
    selfplay_s: float = 0.0
    train_s: float = 0.0
    loss_total: float = 0.0
    loss_policy: float = 0.0
    loss_value: float = 0.0
    loss_l2: float = 0.0
    v_kl: float = 0.0
    top1: float = 0.0
    p_kl: float = 0.0
    grad_rel: float = 0.0
    update_rel: float = 0.0
    p_kl_head: float = 0.0
    p_kl_mid: float = 0.0
    p_kl_tail: float = 0.0
    v_kl_head: float = 0.0
    v_kl_mid: float = 0.0
    v_kl_tail: float = 0.0
    entropy: float = 0.0
    p0_pct: float = 0.0
    p1_pct: float = 0.0
    draw_pct: float = 0.0


# ---------------------------------------------------------------------------
# 2. Parsing
# ---------------------------------------------------------------------------

# First line: step header + throughput
_LINE1_RE = re.compile(
    r"\[step\s+(?P<step>\d+)/\d+\]\s+"
    r"games=\s*(?P<games>\d+)\s+"
    r"states=\s*(?P<states>\d+)\s+"
    r"buffer=\s*(?P<buf_cur>\d+)/\s*(?P<buf_total>\d+)\s+"
    r"unique=\s*(?P<unique>\d+)\s+"
    r"tags=\{(?P<tags>[^}]*)\}\s+"
    r"weak=rare=(?P<rare>\d+)\s+(rf=(?P<rf>\d+)\s+)?"
    r"wf=(?P<wf>\d+)\s+weak=(?P<weak>\d+)\s+"
    r"\(\d+d games\)\s+\|\s+"
    r"rare/g=(?P<rare_g>[\d.]+)\s+(rf/g=(?P<rf_g>[\d.]+)\s+)?"
    r"wf/g=(?P<wf_g>[\d.]+)\s+weak/g=(?P<weak_g>[\d.]+)\s+"
    r"states/s=(?P<sps>[\d.]+)\s+"
    r"selfplay=(?P<selfplay>[\d.]+)s\s+"
    r"train=(?P<train>[\d.]+)s"
)

# Second line: loss + entropy + outcomes
_LINE2_RE = re.compile(
    r"loss=Losses\(total:\s*(?P<loss_total>[\d.]+),\s*"
    r"policy:\s*(?P<loss_policy>[\d.]+),\s*"
    r"value:\s*(?P<loss_value>[\d.]+),\s*"
    r"l2:\s*(?P<loss_l2>[\d.]+)"
    r"(?:,\s*V-KL:\s*(?P<vkl>[\d.]+))?"
    r"(?:,\s*Top1:\s*(?P<top1>[\d.]+)%)?"
    r"(?:,\s*P-KL:\s*(?P<pkl>[\d.]+))?"
    r"(?:,\s*GradRel:\s*(?P<grad_rel>[\d.]+))?"
    r"(?:,\s*UpdRel:\s*(?P<update_rel>[\d.]+))?"
    r"\)\s+"
    r"entropy=(?P<entropy>[\d.]+)\s+\|\s+"
    r"(?:pkl_h=(?P<pkl_h>[\d.]+)\s+)?"
    r"(?:pkl_m=(?P<pkl_m>[\d.]+)\s+)?"
    r"(?:pkl_t=(?P<pkl_t>[\d.]+)\s+)?"
    r"(?:vkl_h=(?P<vkl_h>[\d.]+)\s+)?"
    r"(?:vkl_m=(?P<vkl_m>[\d.]+)\s+)?"
    r"(?:vkl_t=(?P<vkl_t>[\d.]+)\s+)?"
    r"p0=(?P<p0>[\d.]+)%\s+"
    r"p1=(?P<p1>[\d.]+)%\s+"
    r"draw=(?P<draw>[\d.]+)%"
)

# Checkpoint lines
_CKPT_RE = re.compile(r"\[checkpoint\]\s+Saved.*checkpoint-(?P<step>\d+)\.pt")


def parse_training_log(log_path: str) -> List[StepRecord]:
    """Parse a train.log file, returning one StepRecord per step."""
    records: List[StepRecord] = []
    pending: Optional[dict] = None  # step line waiting for its loss line

    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m1 = _LINE1_RE.search(line)
            if m1:
                d = m1.groupdict()
                # Parse tags dict:  {'': 1495, 'rare': 129}
                tags_normal = 0
                tags_rare = 0
                raw_tags = d["tags"]
                for part in raw_tags.split(","):
                    part = part.strip()
                    if not part:
                        continue
                    kv = part.split(":", 1)
                    if len(kv) != 2:
                        continue
                    k = kv[0].strip().strip("'")
                    v = int(kv[1].strip())
                    if k == "":
                        tags_normal = v
                    elif k == "rare":
                        tags_rare = v

                pending = {
                    "step": int(d["step"]),
                    "games": int(d["games"]),
                    "states": int(d["states"]),
                    "buffer_current": int(d["buf_cur"]),
                    "buffer_total": int(d["buf_total"]),
                    "unique_states": int(d["unique"]),
                    "tags_normal": tags_normal,
                    "tags_rare": tags_rare,
                    "rare_count": int(d["rare"]),
                    "rf_count": int(d.get("rf") or 0),
                    "wf_count": int(d["wf"]),
                    "weak_count": int(d["weak"]),
                    "rare_per_game": float(d["rare_g"]),
                    "rf_per_game": float(d.get("rf_g") or 0),
                    "wf_per_game": float(d["wf_g"]),
                    "weak_per_game": float(d["weak_g"]),
                    "states_per_s": float(d["sps"]),
                    "selfplay_s": float(d["selfplay"]),
                    "train_s": float(d["train"]),
                }
                continue

            m2 = _LINE2_RE.search(line)
            if m2 and pending is not None:
                d2 = m2.groupdict()
                r = StepRecord(
                    **pending,
                    loss_total=float(d2["loss_total"]),
                    loss_policy=float(d2["loss_policy"]),
                    loss_value=float(d2["loss_value"]),
                    loss_l2=float(d2["loss_l2"]),
                    v_kl=float(d2.get("vkl") or 0),
                    top1=float(d2.get("top1") or 0) / 100,
                    p_kl=float(d2.get("pkl") or 0),
                    grad_rel=float(d2.get("grad_rel") or 0),
                    update_rel=float(d2.get("update_rel") or 0),
                    p_kl_head=float(d2.get("pkl_h") or 0),
                    p_kl_mid=float(d2.get("pkl_m") or 0),
                    p_kl_tail=float(d2.get("pkl_t") or 0),
                    v_kl_head=float(d2.get("vkl_h") or 0),
                    v_kl_mid=float(d2.get("vkl_m") or 0),
                    v_kl_tail=float(d2.get("vkl_t") or 0),
                    entropy=float(d2["entropy"]),
                    p0_pct=float(d2["p0"]),
                    p1_pct=float(d2["p1"]),
                    draw_pct=float(d2["draw"]),
                )
                records.append(r)
                pending = None

    return records


# ---------------------------------------------------------------------------
# 3. Moving average
# ---------------------------------------------------------------------------

def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    # Pad at the beginning so output length matches input
    padded = np.empty_like(values)
    padded[: window - 1] = values[: window - 1]
    padded[window - 1:] = smoothed
    return padded


# ---------------------------------------------------------------------------
# 4. Plotting
# ---------------------------------------------------------------------------

def plot_metrics(records: List[StepRecord], smooth: int = 1,
                 output_path: Optional[str] = None,
                 elo_pairs: Optional[list] = None):
    """Produce a 3x4 multi-panel dashboard of training metrics."""
    steps = np.array([r.step for r in records])
    if len(steps) < 2:
        print("[metrics] Not enough data to plot.")
        return

    def arr(name: str):
        return np.array([getattr(r, name) for r in records])

    ckpt_mask = np.array([r.step % 5 == 0 for r in records])

    fig, axes = plt.subplots(3, 4, figsize=(16, 9))
    fig.suptitle("Training Metrics Dashboard", fontsize=14, fontweight="bold")

    # ── (0,0) Total loss + L2 ───────────────────────────────────────────
    ax = axes[0, 0]
    t = arr("loss_total")
    l2 = arr("loss_l2")
    if smooth > 1:
        t = moving_average(t, smooth)
        l2 = moving_average(l2, smooth)
    ax.plot(steps, t, color="#333333", linewidth=1.2, label="total")
    ax.set_ylabel("total", color="#333333")
    ax.tick_params(axis="y", colors="#333333")
    ax2 = ax.twinx()
    ax2.plot(steps, l2, color="#9E9E9E", linewidth=1.0, alpha=0.7, label="L2")
    ax2.set_ylabel("L2", color="#9E9E9E")
    ax2.tick_params(axis="y", colors="#9E9E9E")
    ax.set_title("Total Loss + L2")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.25)

    # ── (0,1) Policy + Value loss ───────────────────────────────────────
    ax = axes[0, 1]
    pl = arr("loss_policy")
    vl = arr("loss_value")
    if smooth > 1:
        pl = moving_average(pl, smooth)
        vl = moving_average(vl, smooth)
    ax.plot(steps, pl, color="#2196F3", linewidth=1.2, label="policy")
    ax.plot(steps, vl, color="#FF5722", linewidth=1.2, label="value")
    ax.set_title("Policy / Value Loss")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.25)

    # ── (0,2) Entropy ──────────────────────────────────────────────────
    ax = axes[0, 2]
    ent = arr("entropy")
    if smooth > 1:
        ent = moving_average(ent, smooth)
    ax.plot(steps, ent, color="#4CAF50", linewidth=1.2)
    ax.set_title("Policy Entropy")
    # ax.set_ylim(0.8, 1.8)
    ax.grid(True, alpha=0.25)
    ax.axhline(y=np.log(65), color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

    # ── (0,3) Throughput ───────────────────────────────────────────────
    ax = axes[0, 3]
    sps = arr("states_per_s")
    if smooth > 1:
        sps = moving_average(sps, smooth)
    ax.plot(steps, sps, color="#9C27B0", linewidth=1.2)
    ax.set_title("Throughput (states/s)")
    ax.grid(True, alpha=0.25)

    # ── (1,0) Game outcomes ────────────────────────────────────────────
    ax = axes[1, 0]
    p0 = arr("p0_pct")
    p1 = arr("p1_pct")
    draw = arr("draw_pct")
    if smooth > 1:
        p0 = moving_average(p0, smooth)
        p1 = moving_average(p1, smooth)
        draw = moving_average(draw, smooth)
    ax.fill_between(steps, 0, p0, alpha=0.4, color="#2196F3", label="p0")
    ax.fill_between(steps, p0, p0 + p1, alpha=0.4, color="#FF5722", label="p1")
    ax.fill_between(steps, p0 + p1, 100, alpha=0.4, color="#9E9E9E", label="draw")
    ax.set_title("Game Outcomes")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.25)

    # ── (1,1) Policy quality ───────────────────────────────────────────
    ax = axes[1, 1]
    t1 = arr("top1")
    pkh = arr("p_kl_head")
    pkm = arr("p_kl_mid")
    pkt = arr("p_kl_tail")
    if smooth > 1:
        t1 = moving_average(t1, smooth)
        pkh = moving_average(pkh, smooth)
        pkm = moving_average(pkm, smooth)
        pkt = moving_average(pkt, smooth)
    ax.plot(steps, t1 * 100, color="#2196F3", linewidth=1.2, label="Top1%")
    ax.set_ylabel("Top1 %", color="#2196F3")
    ax.tick_params(axis="y", colors="#2196F3")
    ax2 = ax.twinx()
    ax2.plot(steps, pkh, color="#4CAF50", linewidth=1.0, alpha=0.8, label="P-KL h")
    ax2.plot(steps, pkm, color="#FF9800", linewidth=1.0, alpha=0.8, label="P-KL m")
    ax2.plot(steps, pkt, color="#F44336", linewidth=1.0, alpha=0.8, label="P-KL t")
    ax2.set_ylabel("P-KL", color="#333333")
    ax2.tick_params(axis="y", colors="#333333")
    ax.set_title("Policy Quality")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.25)

    # ── (1,2) Weak-move + rare_flip stats ──────────────────────────────
    ax = axes[1, 2]
    for key, label, color in [
        ("rare_per_game", "rare/g", "#E91E63"),
        ("rf_per_game", "rare_flip/g", "#795548"),
        ("wf_per_game", "weak_final/g", "#FF9800"),
        ("weak_per_game", "weak/g", "#00BCD4"),
    ]:
        v = arr(key)
        if smooth > 1:
            v = moving_average(v, smooth)
        ax.plot(steps, v, label=label, color=color, linewidth=1.0, alpha=0.85)
    ax.set_title("Weak-move + Rare-flip")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.25)

    # ── (1,3) Weight update health ─────────────────────────────────────
    ax = axes[1, 3]
    gr = arr("grad_rel")
    ur = arr("update_rel")
    if smooth > 1:
        gr = moving_average(gr, smooth)
        ur = moving_average(ur, smooth)
    ax.plot(steps, gr, color="#2196F3", linewidth=1.2, label="||∇W||/||W||")
    ax.plot(steps, ur, color="#FF5722", linewidth=1.2, label="||ΔW||/||W||")
    ax.set_title("Weight Update")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)

    # ── (2,0) Buffer health ────────────────────────────────────────────
    ax = axes[2, 0]
    buf_cur = arr("buffer_current")
    buf_max = np.max(buf_cur)
    unique = arr("unique_states")
    unique_ratio = unique / np.maximum(buf_cur, 1) * 100
    ax.plot(steps, buf_cur, color="#607D8B", linewidth=1.2, label="buffer")
    ax2 = ax.twinx()
    ax2.plot(steps, unique_ratio, color="#FF9800", linewidth=1.0, alpha=0.7)
    ax2.set_ylabel("Unique %", color="#FF9800")
    ax2.tick_params(axis="y", colors="#FF9800")
    ax2.set_ylim(0, 105)
    ax.set_title("Buffer Health")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.25)

    # ── (2,1) Value quality ────────────────────────────────────────────
    ax = axes[2, 1]
    vl = arr("loss_value")
    vkh = arr("v_kl_head")
    vkm = arr("v_kl_mid")
    vkt = arr("v_kl_tail")
    if smooth > 1:
        vl = moving_average(vl, smooth)
        vkh = moving_average(vkh, smooth)
        vkm = moving_average(vkm, smooth)
        vkt = moving_average(vkt, smooth)
    ax.plot(steps, vl, color="#FF5722", linewidth=1.2, label="CE loss")
    ax.set_ylabel("Value CE", color="#FF5722")
    ax.tick_params(axis="y", colors="#FF5722")
    ax2 = ax.twinx()
    ax2.plot(steps, vkh, color="#4CAF50", linewidth=1.0, alpha=0.8, label="V-KL h")
    ax2.plot(steps, vkm, color="#FF9800", linewidth=1.0, alpha=0.8, label="V-KL m")
    ax2.plot(steps, vkt, color="#F44336", linewidth=1.0, alpha=0.8, label="V-KL t")
    ax2.set_ylabel("V-KL", color="#333333")
    ax2.tick_params(axis="y", colors="#333333")
    ax.set_title("Value Quality")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.25)

    # ── (2,2) Elo vs step ──────────────────────────────────────────────
    ax = axes[2, 2]
    if elo_pairs:
        elo_steps, elo_vals = zip(*elo_pairs)
        ax.plot(elo_steps, elo_vals, "o-", color="#2196F3", markersize=5,
                linewidth=1.2)
        ax.axhline(y=1000, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.grid(True, alpha=0.3)
        ax.set_title("Elo Rating")
    else:
        ax.text(0.5, 0.5, "no eval data", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        ax.set_title("Elo Rating")

    for ax_row in axes.flat:
        ax_row.set_xlabel("step")
        ax_row.set_xlim(steps[0], steps[-1])

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[metrics] Plot saved to {output_path}")
    plt.show()


# ---------------------------------------------------------------------------
# 5. Summary table
# ---------------------------------------------------------------------------

def print_summary(records: List[StepRecord]):
    """Print summary statistics for key metrics."""
    if not records:
        return

    print("\n=== Training Metrics Summary ===")
    print(f"  Steps: {records[0].step} → {records[-1].step}  ({len(records)} total)")
    total_games = sum(r.games for r in records)
    total_states = sum(r.states for r in records)
    total_hours = sum(r.selfplay_s + r.train_s for r in records) / 3600
    print(f"  Total games: {total_games}  |  Total states: {total_states}  |  Wall time: {total_hours:.1f}h")

    metrics = [
        ("loss_total", "Total loss", "{:.3f} → {:.3f}"),
        ("loss_policy", "Policy loss", "{:.3f} → {:.3f}"),
        ("loss_value", "Value loss", "{:.3f} → {:.3f}"),
        ("v_kl", "V-KL", "{:.4f} → {:.4f}"),
        ("p_kl", "P-KL", "{:.4f} → {:.4f}"),
        ("top1", "Top1 acc", "{:.1%} → {:.1%}"),
        ("entropy", "Entropy", "{:.3f} → {:.3f}"),
        ("states_per_s", "States/s", "{:.0f} → {:.0f}"),
        ("p0_pct", "p0 win%", "{:.1f} → {:.1f}"),
        ("p1_pct", "p1 win%", "{:.1f} → {:.1f}"),
        ("draw_pct", "Draw%", "{:.1f} → {:.1f}"),
        ("unique_states", "Unique states", "{:.0f} → {:.0f}"),
        ("rare_per_game", "Rare/g", "{:.2f} → {:.2f}"),
        ("weak_per_game", "Weak/g", "{:.2f} → {:.2f}"),
        ("grad_rel", "||∂W||/||W||", "{:.6f} → {:.6f}"),
        ("update_rel", "||ΔW||/||W||", "{:.6f} → {:.6f}"),
    ]

    # Use first/last 5% to reduce noise
    n_head = max(1, len(records) // 20)
    head = records[:n_head]
    tail = records[-n_head:]

    print(f"\n  {'Metric':<18s} {'Start':>10s} {'End':>10s} {'Δ':>10s}")
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*10}")
    for attr, label, fmt in metrics:
        v0 = float(np.mean([getattr(r, attr) for r in head]))
        v1 = float(np.mean([getattr(r, attr) for r in tail]))
        print(f"  {label:<18s} {fmt.format(v0, v1):>10s}  {v1-v0:+10.3f}")

    # Weak-move totals
    total_rare = sum(r.rare_count for r in records)
    total_rf = sum(r.rf_count for r in records)
    total_wf = sum(r.wf_count for r in records)
    total_weak = sum(r.weak_count for r in records)
    total_tags_rare = sum(r.tags_rare for r in records)
    total_tags_normal = sum(r.tags_normal for r in records)
    print(f"\n  Weak-move totals: rare={total_rare}  rare_flip={total_rf}  "
          f"weak_final={total_wf}  weak={total_weak}")
    print(f"  Buffer tags: normal={total_tags_normal}  rare_tagged={total_tags_rare}  "
          f"rare_pct={total_tags_rare/max(1,total_tags_normal+total_tags_rare)*100:.1f}%")


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Training metrics dashboard from AlphaZero train.log")
    parser.add_argument("--log", type=str, default=DEFAULT_LOG,
                        help="Path to train.log")
    parser.add_argument("--smooth", type=int, default=DEFAULT_SMOOTH,
                        help=f"Moving-average window (default: {DEFAULT_SMOOTH})")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save the plot (default: <log>_metrics.png)")
    parser.add_argument("--no-plot", action="store_true", default=DEFAULT_NO_PLOT,
                        help="Print table only, skip plot")
    args = parser.parse_args()

    # Auto-detect log path
    log_path = args.log
    if log_path is not None and not Path(log_path).exists():
        print(f"[metrics] Given log not found: {log_path}, trying auto-detect...")
        log_path = None
    if log_path is None:
        print("[metrics] No log file found. Specify --log PATH.")
        sys.exit(1)

    records = parse_training_log(log_path)
    if not records:
        print("[metrics] No training step lines found in log.")
        sys.exit(1)

    print(f"[metrics] Parsed {len(records)} steps from {log_path}")

    print_summary(records)

    if not args.no_plot:
        # Compute Elo from eval lines in the same log
        elo_pairs = None
        try:
            from experiment.elo_analysis import (
                parse_eval_log, build_matchup_dict, simulate_elo)
            eval_records = parse_eval_log(log_path)
            if eval_records:
                matchups = build_matchup_dict(eval_records)
                if matchups:
                    traj = simulate_elo(matchups, n_rounds=5000,
                                        K=16, games_per_pair=100, seed=42)
                    elo_pairs = []
                    for m in sorted(traj.keys(),
                                    key=lambda x: (0 if x == "random"
                                                   else int(x[4:]))):
                        if m == "random":
                            continue
                        step = int(m[4:])
                        elo_pairs.append((step, traj[m][-1]))
                    print(f"[metrics] Elo: {len(elo_pairs)} models")
        except Exception as e:
            print(f"[metrics] Elo computation skipped: {e}")

        output_path = args.output or str(Path(log_path).with_suffix("")) + "_metrics.png"
        plot_metrics(records, smooth=args.smooth, output_path=output_path,
                     elo_pairs=elo_pairs)

DEFAULT_LOG = r"C:\Users\shouk\xiangqi_train\cloud_b\train.log"
DEFAULT_SMOOTH = 1
DEFAULT_NO_PLOT = False

if __name__ == "__main__":
    main()
