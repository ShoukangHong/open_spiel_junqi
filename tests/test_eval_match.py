"""Tests for eval_match: model loading + match execution."""

import json
import os
import tempfile

import numpy as np
import pyspiel
import torch

from train.eval_match import run_match, _model_for, _models


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


# ── _model_for checkpoint switching ────────────────────────────────────────


def _save_checkpoint(model, step, tmpdir):
    """Save a checkpoint file — uses model's own optimizer state."""
    torch.save(
        {"model_state_dict": model._model.state_dict(),
         "optimizer_state_dict": model._optimizer.state_dict(),
         "step": step},
        os.path.join(tmpdir, f"checkpoint-{step}.pt"))


def test_model_for_different_steps_different_output():
    """_model_for(step=A) and _model_for(step=B) differ when A != B."""
    game = pyspiel.load_game("othello")
    from train.core.model_builder import build_othello_model

    cfg = {"nn_width": 8, "nn_depth": 2, "device": "cpu",
           "learning_rate": 1e-3, "weight_decay": 1e-4, "game": "othello"}

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg["path"] = tmpdir
        with open(os.path.join(tmpdir, "train_config.json"), "w") as f:
            json.dump(cfg, f)

        # Checkpoint 100 — random weights seed 42
        torch.manual_seed(42)
        m = build_othello_model(game, cfg)
        _save_checkpoint(m, 100, tmpdir)

        # Checkpoint 200 — random weights seed 99 (different)
        torch.manual_seed(99)
        m2 = build_othello_model(game, cfg)
        _save_checkpoint(m2, 200, tmpdir)

        _models.clear()

        # Non-zero observation (random board position)
        rng = np.random.RandomState(7)
        obs = rng.randn(1, *game.observation_tensor_shape()).astype(np.float32)
        mask = np.ones((1, game.num_distinct_actions()), dtype=bool)

        model_100 = _model_for({"checkpoint_dir": tmpdir, "checkpoint_step": 100})
        model_200 = _model_for({"checkpoint_dir": tmpdir, "checkpoint_step": 200})

        # Sanity: weights should differ after loading different checkpoints
        w100_list = [p.data.clone() for p in model_100._model.parameters()]
        w200_list = [p.data.clone() for p in model_200._model.parameters()]
        all_same = all(torch.allclose(a, b, atol=1e-8)
                       for a, b in zip(w100_list, w200_list))
        assert not all_same, \
            "model weights should differ after loading different checkpoints"

        val_100, pol_100 = model_100.inference(obs, mask)
        val_200, pol_200 = model_200.inference(obs, mask)

        assert not np.allclose(pol_100, pol_200, atol=0.01), \
            f"policy should differ: step 100 vs step 200"


def test_model_for_same_step_returns_cached():
    """_model_for returns the same object for the same (dir, step)."""
    game = pyspiel.load_game("othello")
    from train.core.model_builder import build_othello_model

    cfg = {"nn_width": 8, "nn_depth": 2, "device": "cpu",
           "learning_rate": 1e-3, "weight_decay": 1e-4, "game": "othello"}

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg["path"] = tmpdir
        with open(os.path.join(tmpdir, "train_config.json"), "w") as f:
            json.dump(cfg, f)

        m = build_othello_model(game, cfg)
        _save_checkpoint(m, 50, tmpdir)

        _models.clear()

        pcfg = {"checkpoint_dir": tmpdir, "checkpoint_step": 50}
        a = _model_for(pcfg)
        b = _model_for(pcfg)
        assert a is b, "_model_for should cache and return the same object"


def test_model_for_sequential_builds_match_eval_flow():
    """Simulates the eval loop: cur + 3 refs with different random seeds."""
    game = pyspiel.load_game("othello")
    from train.core.model_builder import build_othello_model

    cfg = {"nn_width": 8, "nn_depth": 2, "device": "cpu",
           "learning_rate": 1e-3, "weight_decay": 1e-4, "game": "othello"}

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg["path"] = tmpdir
        with open(os.path.join(tmpdir, "train_config.json"), "w") as f:
            json.dump(cfg, f)

        cur_step = 300

        # Save current checkpoint (seed 7)
        torch.manual_seed(7)
        mc = build_othello_model(game, cfg)
        _save_checkpoint(mc, cur_step, tmpdir)

        # Save 3 ref checkpoints with different random seeds
        for step, seed in [(60, 10), (120, 20), (250, 30)]:
            torch.manual_seed(seed)
            mr = build_othello_model(game, cfg)
            _save_checkpoint(mr, step, tmpdir)

        _models.clear()

        rng = np.random.RandomState(1)
        obs = rng.randn(1, *game.observation_tensor_shape()).astype(np.float32)
        mask = np.ones((1, game.num_distinct_actions()), dtype=bool)

        # Current model
        cur_model = _model_for({"checkpoint_dir": tmpdir,
                                "checkpoint_step": cur_step})
        _, cur_pol = cur_model.inference(obs, mask)

        ref_policies = {}
        for step in [60, 120, 250]:
            ref_model = _model_for({"checkpoint_dir": tmpdir,
                                    "checkpoint_step": step})
            _, ref_pol = ref_model.inference(obs, mask)
            ref_policies[step] = ref_pol

        # Every ref should differ from current
        for step, ref_pol in ref_policies.items():
            assert not np.allclose(cur_pol, ref_pol, atol=0.01), \
                f"ref step {step} should differ from cur step {cur_step}"

        # Refs should also differ from each other
        steps = list(ref_policies.keys())
        for i in range(len(steps)):
            for j in range(i + 1, len(steps)):
                assert not np.allclose(ref_policies[steps[i]],
                                       ref_policies[steps[j]], atol=0.01), \
                    f"ref {steps[i]} and {steps[j]} should differ"
