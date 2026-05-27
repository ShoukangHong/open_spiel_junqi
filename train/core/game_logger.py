"""Per-actor game logger."""

import os
import time


class GameLogger:
    """Writes one file per actor in the checkpoint dir."""

    def __init__(self, log_dir: str, actor_id: int, sample_rate: int = 100):
        os.makedirs(log_dir, exist_ok=True)
        self._path = os.path.join(log_dir, f"actor_{actor_id}.log")
        self._file = open(self._path, "a", encoding="utf-8", buffering=1)
        self._game_count = 0
        self._sample_rate = sample_rate
        self._verbose_game = False

    def log_game_start(self, weak_side=None):
        self._game_count += 1
        self._verbose_game = (self._game_count % self._sample_rate == 1)
        self._summary = []
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        if self._verbose_game:
            parts = [f"\n{'=' * 60}",
                     f"Game {self._game_count}  {ts}"]
            if weak_side is not None:
                parts.append(f"  weak_side = {weak_side}")
            self._write("\n".join(parts))
        self._summary.append(f"Game {self._game_count}  {ts}")
        if weak_side is not None:
            self._summary.append(f" weak_side={weak_side}")

    def log_move(self, move_num, player, state, mcts_top5, chosen_action,
                 tag="", tau=1.0):
        if self._verbose_game:
            pname = "BLACK(p0)" if player == 0 else "WHITE(p1)"
            extra = f"  [{tag}]" if tag else ""
            self._write(
                f"\n── Move {move_num}  {pname}  tau={tau:.2f}{extra} ──\n"
                f"{state}\n"
                f"MCTS top-5: {mcts_top5}\n"
                f"Chosen: {chosen_action}"
            )

    def log_game_end(self, returns, move_count, rare_games=0):
        if self._verbose_game:
            self._write(
                f"\n── Final (move {move_count}) ──\n"
                f"Returns: {returns[0]:+.0f}/{returns[1]:+.0f}"
                f"  rare_games: {rare_games}"
            )
        else:
            self._summary.append(
                f" moves={move_count} "
                f"ret={returns[0]:+.0f}/{returns[1]:+.0f}"
                f" rare_games={rare_games}")
            self._write("  ".join(self._summary))

    def log_line(self, msg: str):
        self._write(msg)

    def _write(self, text: str):
        self._file.write(text + "\n")
        self._file.flush()

    def close(self):
        self._file.close()
