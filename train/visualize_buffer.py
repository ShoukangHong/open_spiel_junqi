"""Visualize buffer samples: board, current player, policy, value.

Usage:
    python visualize_buffer.py                        # show all samples
    python visualize_buffer.py --index 42             # show sample 42
    python visualize_buffer.py --file my_buffer.npz   # custom file
"""

import argparse
import os

import numpy as np

BUFFER_FILE = r"C:\Users\shouk\othello_train_v2\buffer-checkpoint-45.npz"

COLS = 8


def board_str(obs):
    """Convert [4, 8, 8] observation to human-readable board string."""
    obs = np.reshape(obs, (4, COLS, COLS)).astype(int)
    # Determine current player from channel 3
    # Channel 3: all 1 = black to move, all 0 = white to move
    cur = 0 if obs[3, 0, 0] == 1 else 1
    player_name = "BLACK(p0)" if cur == 0 else "WHITE(p1)"

    lines = []
    lines.append(f"  Turn: {player_name}")
    lines.append("  " + "-" * (COLS * 2 + 1))
    for row in range(COLS):
        cells = []
        for col in range(COLS):
            if obs[0, row, col]:
                cells.append(".")
            elif obs[1, row, col]:
                cells.append("X")
            elif obs[2, row, col]:
                cells.append("O")
            else:
                cells.append("?")  # shouldn't happen
        lines.append(f"  |{' '.join(cells)}|")
    lines.append("  " + "-" * (COLS * 2 + 1))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Visualize buffer samples")
    parser.add_argument("--file", default=BUFFER_FILE, help="Path to .npz file")
    parser.add_argument("--index", type=int, default=-1,
                        help="Show specific sample index (default: show all)")
    args = parser.parse_args()

    data = np.load(args.file)
    obs_all = data["obs"]
    masks = data["masks"]
    policies = data["policies"]
    values = data["values"]
    N = len(values)
    print(f"Loaded {args.file}")
    print(f"  samples: {N}\n")

    indices = [args.index] if args.index >= 0 and args.index < N else range(N)

    for i in indices:
        obs = obs_all[i]
        mask = masks[i]
        policy = policies[i]
        value = values[i]

        # --- Board ---
        print(board_str(obs))

        # --- Value ---
        print(f"  Value: {value:+.3f}")

        # --- Policy (top-N legal actions) ---
        legal = np.where(mask)[0]
        legal_policy = sorted(
            [(a, policy[a]) for a in legal],
            key=lambda x: -x[1])

        # Map action to board coordinate
        policy_lines = []
        for a, p in legal_policy[:10]:  # top 10
            if a >= 64:
                coord = "pass"
            else:
                row, col = a // COLS, a % COLS
                coord = f"({row},{col})"
            bar = "█" * int(p * 50)
            policy_lines.append(f"    {coord:>6s} a{a:>3d}: {p:.3f} {bar}")
        print(f"  Policy (top {min(10, len(legal_policy))}/{len(legal)}):")
        print("\n".join(policy_lines))

        if args.index < 0 and i < N - 1:
            print(f"\n{'─' * 60}")
        else:
            print()


if __name__ == "__main__":
    main()
