import numpy as np

from causalstate.world.envs import make_env, pruned_obs
from causalstate.world.gridworld import (
    CAUSAL_ORACLE,
    VAR_SLICES,
    FactoredGridWorld,
)


def test_make_env():
    env = make_env(start_seed=2)

    assert isinstance(env, FactoredGridWorld)
    assert env._door_pos == (3, 2)


def test_pruned_shape():
    env = FactoredGridWorld()
    env = pruned_obs(env, CAUSAL_ORACLE)

    obs, _ = env.reset(seed=2)

    assert obs.shape == (7,)
    assert env.observation_space.contains(obs)


def test_pruned_values():
    full = FactoredGridWorld()
    wrapped = pruned_obs(FactoredGridWorld(), CAUSAL_ORACLE)

    full_obs, _ = full.reset(seed=2)
    wrapped_obs, _ = wrapped.reset(seed=2)

    expected = np.concatenate(
        [
            full_obs[VAR_SLICES["s0"]],
            full_obs[VAR_SLICES["s1"]],
            full_obs[VAR_SLICES["s2"]],
            full_obs[VAR_SLICES["s3"]],
        ]
    )

    np.testing.assert_array_equal(
        wrapped_obs,
        expected,
    )


def test_pruned_delegates_do_release():
    env = pruned_obs(FactoredGridWorld(), CAUSAL_ORACLE)

    env.reset(seed=2)

    env.do("s2", 1, hold=True)

    obs, _ = env.reset(seed=2)

    assert obs[4] == 0.0

    env.do("s2", 1)

    obs = env.unwrapped.extract_state()

    assert obs[4] == 1.0

    env.release("s2")