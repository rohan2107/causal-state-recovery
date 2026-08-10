import warnings
from pathlib import Path

import numpy as np
import pytest

from causalstate.evaluation.metrics import select
from causalstate.observational.baselines import (
    SUMMARY_VARS,
    correlation_scores,
    episode_summaries,
    mi_scores,
    run_baselines,
)
from causalstate.observational.dataset import load_split

DATASET = Path("data/observational/train.npz")

def make_split(obs, y, episode_id):

    return {
        "obs": np.asarray(obs, dtype=np.float32),
        "Y": np.asarray(y, dtype=np.int8),
        "episode_id": np.asarray(
            episode_id,
            dtype=np.int32,
        ),
    }

def test_episode_summaries_shape_and_grouping():

    obs = np.zeros((6, 9), dtype=np.float32)

    # episode 0
    obs[0, 3] = 0
    obs[1, 3] = 2

    # episode 1
    obs[2, 3] = 1
    obs[3, 3] = 1
    obs[4, 3] = 1

    # episode 2
    obs[5, 3] = 5

    split = make_split(
        obs,
        [1, 1, 0, 0, 0, 1],
        [0, 0, 1, 1, 1, 2],
    )

    X, y = episode_summaries(split)

    assert X.shape == (3, len(SUMMARY_VARS))
    assert y.shape == (3,)

    np.testing.assert_allclose(
        X[:, 0],
        [1.0, 1.0, 5.0],
    )

def test_episode_summaries_y_is_per_episode():

    obs = np.zeros((4, 9), dtype=np.float32)

    split = make_split(
        obs,
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    )

    _, y = episode_summaries(split)

    np.testing.assert_array_equal(
        y,
        [1, 0],
    )

def test_episode_summaries_use_correct_variable_slices():

    obs = np.zeros((2, 9), dtype=np.float32)

    obs[:, 8] = 7

    split = make_split(
        obs,
        [1, 1],
        [0, 0],
    )

    X, _ = episode_summaries(split)

    np.testing.assert_allclose(
        X[0],
        [0, 0, 0, 7],
    )

def test_correlation_scores_returns_summary_vars():

    rng = np.random.default_rng(0)

    X = rng.random((20, 4))

    y = rng.integers(
        0,
        2,
        size=20,
    )

    scores = correlation_scores(X, y)

    assert set(scores) == set(SUMMARY_VARS)

def test_select_applies_threshold():

    scores = {
        "s1": 0.20,
        "s2": 0.05,
        "s4": 0.30,
        "s5": 0.01,
    }

    assert select(scores, 0.10) == {
        "s1",
        "s4",
    }

def test_mi_scores_returns_summary_vars():

    rng = np.random.default_rng(0)

    X = rng.random((30, 4))

    y = rng.integers(
        0,
        2,
        size=30,
    )

    scores = mi_scores(X, y)

    assert set(scores) == set(SUMMARY_VARS)

    assert all(
        score >= 0
        for score in scores.values()
    )

def test_run_baselines_returns_expected_keys():

    rng = np.random.default_rng(0)

    obs = rng.random((60, 9), dtype=np.float32)

    episode_ids = np.repeat(np.arange(6), 10)

    y = (
        [1] * 20 +
        [0] * 20 +
        [1] * 20
    )

    split = make_split(
        obs,
        y,
        episode_ids,
    )

    result = run_baselines(split)

    assert set(result) == {
        "correlation",
        "mutual_information",
        "corr_selected",
        "mi_selected",
        "corr_recovery",
        "mi_recovery",
        "n_episodes",
    }

    assert set(result["corr_recovery"]) == {
        "precision",
        "recall",
        "f1",
        "true_positives",
        "false_positives",
        "false_negatives",
    }

    assert set(result["mi_recovery"]) == {
        "precision",
        "recall",
        "f1",
        "true_positives",
        "false_positives",
        "false_negatives",
    }

    assert isinstance(
        result["corr_recovery"]["f1"],
        float,
    )

    assert isinstance(
        result["mi_recovery"]["f1"],
        float,
    )

    assert result["n_episodes"] == 6

def test_correlation_scores_constant_column_returns_zero():

    X = np.ones((20, 4), dtype=np.float32)

    rng = np.random.default_rng(0)

    y = rng.integers(
        0,
        2,
        size=20,
    )

    with warnings.catch_warnings():

        warnings.filterwarnings(
            "error",
            category=RuntimeWarning,
        )

        scores = correlation_scores(X, y)

    assert all(
        score == 0.0
        for score in scores.values()
    )

def test_mi_scores_are_deterministic():

    rng = np.random.default_rng(0)

    X = rng.random((40, 4))

    y = rng.integers(
        0,
        2,
        size=40,
    )

    assert mi_scores(X, y) == mi_scores(X, y)

def test_select_excludes_threshold_boundary():

    scores = {
        "s1": 0.10,
        "s2": 0.11,
    }

    assert select(
        scores,
        0.10,
    ) == {"s2"}

def test_correlation_is_absolute():
    y = np.array([0, 0, 0, 1, 1, 1])
    X = np.column_stack([y, 1 - y, y, 1 - y]).astype(np.float64)
    scores = correlation_scores(X, y)
    assert all(s == pytest.approx(1.0) for s in scores.values())


def test_run_baselines_detects_true_signal():

    rng = np.random.default_rng(0)

    n_episodes = 60
    steps = 10

    obs = np.zeros(
        (n_episodes * steps, 9),
        dtype=np.float32,
    )

    episode_ids = np.repeat(
        np.arange(n_episodes),
        steps,
    )

    y = []

    for ep in range(n_episodes):

        label = ep % 2

        y.extend([label] * steps)

        mask = episode_ids == ep

        # s2 perfectly predicts Y
        obs[mask, 4] = label

        # s4 is pure noise
        obs[mask, 7] = rng.standard_normal(steps)

    split = make_split(
        obs,
        y,
        episode_ids,
    )

    result = run_baselines(split)

    assert "s2" in result["corr_selected"]
    assert "s4" not in result["corr_selected"]

@pytest.mark.data
@pytest.mark.skipif(
    not DATASET.exists(),
    reason="run scripts/build_dataset.py",
)
def test_baselines_on_train_split():

    result = run_baselines(
        load_split(DATASET)
    )

    assert result["n_episodes"] == 420

    assert result["mi_selected"] == {
        "s1",
        "s2",
        "s4",
        "s5",
    }

    assert "s4" not in result["corr_selected"]

    assert result["correlation"]["s1"] > 0.7
