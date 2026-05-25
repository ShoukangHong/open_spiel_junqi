"""Smoke tests for eval_match.run_match (no model loading)."""

import numpy as np
from train.eval_match import run_match


def _rnd(cfg=None):
    return dict(cfg or {}, strategy="random")


def _gdy(cfg=None):
    return dict(cfg or {}, strategy="greedy")


def test_score_sums_to_game_count():
    score, _ = run_match(_rnd(), _gdy(), num_games=30, quiet=True)
    assert sum(score.values()) == 30


def test_sequences_match_game_count():
    _, seqs = run_match(_rnd(), _gdy(), num_games=25, quiet=True)
    assert len(seqs) == 25


def test_random_vs_self():
    score, _ = run_match(_rnd(), _rnd(), num_games=40, quiet=True)
    total = sum(score.values())
    assert total == 40


def test_prefix_diversity():
    """At least some games should have different first moves."""
    _, seqs = run_match(_rnd(), _rnd(), num_games=50, quiet=True)
    first_moves = {s[0] for _, s in seqs if len(s) > 0}
    assert len(first_moves) >= 2, "random should produce varied openings"


def test_top_dog_vs_random():
    """Verify a named stronger strategy is tracked correctly."""
    rnd = _rnd()
    gdy = _gdy()
    score, _ = run_match(rnd, gdy, num_games=10, quiet=True)
    for k in score:
        assert k in ("random", "greedy", "draw"), f"unexpected key {k}"
