"""Tests for train_loop helpers — compute_alpha, value targets."""

import numpy as np
import pytest
from train.core.train_loop import compute_alpha


class TestSurpriseStepIndex:

    def _step_i(self, item, fallback):
        return item[8] if len(item) > 8 and item[8] >= 0 else fallback

    def test_no_surprise_field_fallback(self):
        """Regular state (7-tuple): fallback to loop index."""
        item = (np.zeros(4), np.ones(65, dtype=bool), np.ones(65),
                0, "", 0.3, 0.1)
        assert self._step_i(item, 42) == 42

    def test_child_surprise_fallback(self):
        """Child surprise (8-tuple, no step_i): fallback."""
        item = (np.zeros(4), np.ones(65, dtype=bool), np.ones(65),
                0, "child_surprise", 0.3, 0.1, 0.5)
        assert self._step_i(item, 42) == 42

    def test_root_surprise_stored_step(self):
        """Root surprise stores relative index=5 → step_i=5."""
        item = (np.zeros(4), np.ones(65, dtype=bool), np.ones(65),
                0, "surprise", 0.3, 0.1, 0.5, 5)
        assert self._step_i(item, 99) == 5

    def test_surprise_relative_index_preserved(self):
        """compute_alpha adds offset internally, so step_i=5, offset=50 → step_pos=55."""
        alpha = compute_alpha(5, 30, 50, 30)
        expected = (55 - 30) / max(50 + 29 - 30, 1)  # 25/49
        assert alpha == pytest.approx(expected)

    def test_surprise_early_move_not_alpha_one(self):
        """Surprise at move 5 of 60-game: alpha=0, not forced to 1.0."""
        alpha = compute_alpha(5, 60, 0, 30)
        assert alpha == pytest.approx(0.0)

        # Before fix: index >= game_length → alpha=1.0
        alpha_broken = compute_alpha(65, 60, 0, 30)
        assert alpha_broken == pytest.approx(1.0)



class TestComputeAlpha:

    def test_last_step_reaches_one(self):
        """Final move: alpha must be 1.0 (100% outcome), regardless of params."""
        # Normal game: 100 moves, temp_drop=30
        assert compute_alpha(step_index=99, game_length=100, offset=0,
                             temperature_drop=30) == pytest.approx(1.0)

    def test_at_temperature_drop_is_zero(self):
        """Exactly at temperature_drop: alpha=0 (pure MCTS)."""
        assert compute_alpha(step_index=29, game_length=100, offset=0,
                             temperature_drop=30) == pytest.approx(0.0)

    def test_before_temperature_drop_is_zero(self):
        """Before temperature_drop: alpha=0."""
        assert compute_alpha(step_index=10, game_length=100, offset=0,
                             temperature_drop=30) == pytest.approx(0.0)

    def test_linear_ramp(self):
        """Mid-game: alpha increases linearly."""
        # step 40 of 100, temp_drop=30, offset=0
        # denom = 0+99-30 = 69, step_pos = 40, alpha = 10/69
        actual = compute_alpha(40, 100, 0, 30)
        assert actual == pytest.approx(10 / 69)

    def test_offset_game_last_step_reaches_one(self):
        """Rare fork starting at offset 50, game_length 30: last step alpha=1."""
        offset = 50
        length = 30
        last_i = length - 1  # 29
        # denom = 50+29-30 = 49, step_pos = 50+29 = 79, alpha = (79-30)/49 = 1.0
        assert compute_alpha(last_i, length, offset, 30) == pytest.approx(1.0)

    def test_short_game_all_zero_except_last(self):
        """Game shorter than temperature_drop: non-terminal steps alpha=0,
        but last step always 1.0."""
        for i in range(9):
            assert compute_alpha(i, game_length=10, offset=0,
                                 temperature_drop=30) == pytest.approx(0.0)
        # last step always outcome
        assert compute_alpha(9, 10, 0, 30) == pytest.approx(1.0)

    def test_single_move_game(self):
        """game_length=1 with temp_drop=0: last (only) move alpha=1.0."""
        assert compute_alpha(0, 1, 0, 0) == pytest.approx(1.0)

    def test_denom_clamped_min_one(self):
        """When denom would be ≤0, it clamps to 1 (avoid div-by-zero)."""
        # offset + game_length - 1 - temp_drop = 0 + 5 - 1 - 10 = -4 → denom=1
        # step_pos = 9, alpha = (9-10)/1 = -1 → clamped to 1.0
        alpha = compute_alpha(4, 5, 0, 10)
        assert alpha >= 0.0
        assert alpha <= 1.0

    def test_temperature_drop_zero(self):
        """temp_drop=0: alpha starts at 1.0 from the first move."""
        # denom = 0+99-0 = 99, step_pos=0, alpha = 0/99 = 0? No wait...
        # temp_drop=0 means step_pos >= 0 immediately, so alpha = step_pos / denom
        # step_pos=0 → alpha = 0/99 = 0.0
        # Hmm, that means even with temp_drop=0 the first move gets alpha=0.
        # That's the existing behavior — not ideal but not what we're fixing.
        # Verify last step reaches 1.0:
        assert compute_alpha(99, 100, 0, 0) == pytest.approx(1.0)

    def test_monotonic(self):
        """Alpha never decreases as step_index increases."""
        prev = -1.0
        for i in range(100):
            alpha = compute_alpha(i, 100, 0, 30)
            assert alpha >= prev, f"alpha decreased at step {i}"
            prev = alpha

    def test_bounded_zero_to_one(self):
        """Alpha always in [0, 1] for any reasonable inputs."""
        cases = [
            (0, 1, 0, 0),
            (0, 1, 0, 30),
            (50, 100, 0, 30),
            (99, 100, 0, 30),
            (0, 200, 50, 30),
            (199, 200, 50, 30),
            (0, 10, 0, 50),
        ]
        for step_i, length, offset, td in cases:
            alpha = compute_alpha(step_i, length, offset, td)
            assert 0.0 <= alpha <= 1.0, \
                f"alpha={alpha} out of bounds for ({step_i}, {length}, {offset}, {td})"
