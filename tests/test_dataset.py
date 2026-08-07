from pathlib import Path

import numpy as np
import pytest

from causalstate.observational.dataset import (
    EPS_POOL,
    collect_observational,
    content_hash,
    load_split,
    save_split,
    split_by_episode,
)
from causalstate.world.envs import make_env


def synthetic_dataset() -> dict[str, np.ndarray]:
    episode_id = np.repeat(np.arange(30), 3)

    return {
        "obs": np.zeros((90, 9), dtype=np.float32),
        "Y": (episode_id % 2).astype(np.int8),
        "eps": np.repeat(0.15, 90).astype(np.float32),
        "episode_id": episode_id.astype(np.int32),
        "step": np.tile(np.arange(3), 30).astype(np.int32),
    }

def test_content_hash_is_deterministic():
    split = {
        "obs": np.ones((3, 9), dtype=np.float32),
        "Y": np.array([1, 1, 0], dtype=np.int8),
    }

    assert content_hash(split) == content_hash(split)


def test_content_hash_changes_when_data_changes():
    split1 = {
        "obs": np.ones((3, 9), dtype=np.float32),
        "Y": np.array([1, 1, 0], dtype=np.int8),
    }

    split2 = {
        "obs": np.zeros((3, 9), dtype=np.float32),
        "Y": np.array([1, 1, 0], dtype=np.int8),
    }

    assert content_hash(split1) != content_hash(split2)


def test_save_load_roundtrip(tmp_path: Path):
    split = {
        "obs": np.random.rand(5, 9).astype(np.float32),
        "Y": np.array([1, 0, 1, 1, 0], dtype=np.int8),
        "eps": np.array([0.15, 0.15, 0.30, 0.45, 0.45], dtype=np.float32),
        "episode_id": np.array([0, 0, 1, 1, 1], dtype=np.int32),
        "step": np.array([0, 1, 0, 1, 2], dtype=np.int32),
    }

    path = tmp_path / "split.npz"

    save_split(path, split)

    loaded = load_split(path)

    assert set(loaded) == set(split)

    for key in split:
        np.testing.assert_array_equal(
            loaded[key],
            split[key],
        )

@pytest.mark.slow
def test_collect_observational_schema(trained_model):
    env = make_env(rho=0.7)

    data = collect_observational(
        trained_model,
        env,
        n_episodes=2,
    )

    assert set(data) == {
        "obs",
        "Y",
        "eps",
        "episode_id",
        "step",
    }

    n = len(data["Y"])

    assert n > 0

    assert data["obs"].shape[0] == n
    assert len(data["eps"]) == n
    assert len(data["episode_id"]) == n
    assert len(data["step"]) == n

    env.close()

@pytest.mark.slow
def test_collect_observational_episode_y_constant(trained_model):
    env = make_env(rho=0.7)

    data = collect_observational(
        trained_model,
        env,
        n_episodes=5,
    )

    for episode_id in np.unique(data["episode_id"]):
        mask = data["episode_id"] == episode_id
        assert len(np.unique(data["Y"][mask])) == 1

    env.close()

@pytest.mark.slow
def test_collect_observational_steps_are_sequential(trained_model):
    env = make_env(rho=0.7)

    data = collect_observational(
        trained_model,
        env,
        n_episodes=5,
    )

    for episode_id in np.unique(data["episode_id"]):
        mask = data["episode_id"] == episode_id
        steps = data["step"][mask]

        np.testing.assert_array_equal(
            steps,
            np.arange(len(steps)),
        )

    env.close()

@pytest.mark.slow
def test_collect_observational_eps_from_pool(trained_model):
    env = make_env(rho=0.7)

    data = collect_observational(
        trained_model,
        env,
        n_episodes=20,
    )

    np.testing.assert_allclose(
        np.unique(data["eps"]),
        np.array(EPS_POOL),
    )

    env.close()

def test_split_by_episode_has_no_leakage():
    data = synthetic_dataset()

    splits = split_by_episode(data)

    train = set(np.unique(splits["train"]["episode_id"]))
    val = set(np.unique(splits["val"]["episode_id"]))
    test = set(np.unique(splits["test"]["episode_id"]))

    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)

def test_split_by_episode_preserves_all_episodes():
    data = synthetic_dataset()

    splits = split_by_episode(data)

    original = set(np.unique(data["episode_id"]))

    recovered = (
        set(np.unique(splits["train"]["episode_id"]))
        | set(np.unique(splits["val"]["episode_id"]))
        | set(np.unique(splits["test"]["episode_id"]))
    )

    assert recovered == original

def test_split_sizes_are_reasonable():
    data = synthetic_dataset()

    splits = split_by_episode(data)

    total = len(np.unique(data["episode_id"]))

    train = len(np.unique(splits["train"]["episode_id"]))
    val = len(np.unique(splits["val"]["episode_id"]))
    test = len(np.unique(splits["test"]["episode_id"]))

    assert train + val + test == total

def test_split_preserves_row_alignment():
    data = synthetic_dataset()

    splits = split_by_episode(data)

    for split in splits.values():
        n = len(split["Y"])

        assert split["obs"].shape[0] == n
        assert len(split["eps"]) == n
        assert len(split["episode_id"]) == n
        assert len(split["step"]) == n

@pytest.mark.slow
def test_collect_observational_is_reproducible(trained_model):
    env1 = make_env(rho=0.7)
    env2 = make_env(rho=0.7)

    d1 = collect_observational(
        trained_model,
        env1,
        n_episodes=8,
        start_seed=777,
    )

    d2 = collect_observational(
        trained_model,
        env2,
        n_episodes=8,
        start_seed=777,
    )

    assert content_hash(d1) == content_hash(d2)

    env1.close()
    env2.close()
