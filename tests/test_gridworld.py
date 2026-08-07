import numpy as np
import pytest
from minigrid.core.world_object import Goal, Key, Wall

from causalstate.world.gridworld import (
    ALL_VARS,
    OBS_DIM,
    VAR_SLICES,
    FactoredGridWorld,
    get_var,
)

DEFAULT_LAYOUT_SEED = 2


@pytest.fixture
def env():
    return FactoredGridWorld()

def positions_of(env, cls):
    positions = []

    for x in range(env.width):
        for y in range(env.height):
            if isinstance(env.grid.get(x, y), cls):
                positions.append((x, y))

    return sorted(positions)

def test_observation_slots(env):
    obs, _ = env.reset(seed=DEFAULT_LAYOUT_SEED)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert len(get_var(obs, "s0")) == 3
    assert get_var(obs, "s1").item() == 0.0
    assert get_var(obs, "s2").item() == 0.0
    goal = get_var(obs, "s3")
    assert goal[0] > env._door_pos[0]
    assert 1 <= goal[0] <= 4
    assert 1 <= goal[1] <= 4
    assert np.isfinite(get_var(obs, "s4").item())
    assert get_var(obs, "s5").item() in (0.0, 1.0)
    covered = set()
    for var in ALL_VARS:
        covered.update(range(*VAR_SLICES[var].indices(OBS_DIM)))
    assert covered == set(range(OBS_DIM))

def test_seed2_layout(env):
    env.reset(seed=DEFAULT_LAYOUT_SEED)

    assert env._door_pos == (3, 2)
    assert env._door.is_locked
    assert not env._door.is_open

    assert positions_of(env, Key) == [(1, 4)]
    goals = positions_of(env, Goal)
    assert len(goals) == 1
    goal = goals[0]
    assert goal[0] > env._door_pos[0]
    assert 1 <= goal[1] <= 4
    assert isinstance(env.grid.get(3, 1), Wall)
    assert isinstance(env.grid.get(3, 3), Wall)
    assert isinstance(env.grid.get(3, 4), Wall)

def test_determinism():
    env1 = FactoredGridWorld()
    env2 = FactoredGridWorld()

    actions = [
        env1.actions.forward,
        env1.actions.pickup,
        env1.actions.forward,
        env1.actions.right,
        env1.actions.forward,
    ]

    obs1, _ = env1.reset(seed=7)
    obs2, _ = env2.reset(seed=7)

    trace1 = [obs1.copy()]
    trace2 = [obs2.copy()]

    for action in actions:
        obs1, *_ = env1.step(action)
        obs2, *_ = env2.step(action)

        trace1.append(obs1.copy())
        trace2.append(obs2.copy())

    trace1 = np.stack(trace1)
    trace2 = np.stack(trace2)

    assert np.array_equal(trace1, trace2)

    s4_trace = trace1[:, VAR_SLICES["s4"]].flatten()
    assert len(np.unique(s4_trace)) > 1

def test_scripted_solve(env):
    A = env.actions

    env.reset(seed=DEFAULT_LAYOUT_SEED)

    env.do("s0", (2, 2, 0))
    env.do("s3", (4, 4))

    obs, *_ = env.step(A.toggle)
    assert get_var(obs, "s2").item() == 0.0

    env.do("s1", 1)
    assert get_var(env.extract_state(), "s1").item() == 1.0

    obs, *_ = env.step(A.toggle)
    assert get_var(obs, "s2").item() == 1.0
    assert env._door.is_open

    obs, *_ = env.step(A.forward)
    assert tuple(env.agent_pos) == (3, 2)

    obs, *_ = env.step(A.forward)
    assert tuple(env.agent_pos) == (4, 2)

    obs, *_ = env.step(A.right)
    assert get_var(obs, "s0")[2] == 1.0

    obs, *_ = env.step(A.forward)
    assert tuple(env.agent_pos) == (4, 3)

    obs, reward, terminated, truncated, info = env.step(A.forward)

    assert terminated
    assert info["Y"] == 1
    assert not truncated
    assert reward == 0.98250

def test_hold_clamp(env):
    A = env.actions

    env.reset(seed=DEFAULT_LAYOUT_SEED)
    env.do("s0", (2, 2, 0))
    env.do("s1", 1)
    env.do("s2", 0, hold=True)

    obs, *_ = env.step(A.toggle)
    assert get_var(obs, "s2").item() == 0.0
    assert not env._door.is_open

    obs, *_ = env.step(A.forward)
    assert tuple(env.agent_pos) == (2, 2)

    env.release("s2")

    obs, *_ = env.step(A.toggle)
    assert get_var(obs, "s2").item() == 1.0

def test_key_destruction(env):
    A = env.actions

    env.reset(seed=DEFAULT_LAYOUT_SEED)
    env.do("s0", (1, 3, 1))

    obs, *_ = env.step(A.pickup)
    assert get_var(obs, "s1").item() == 1.0
    assert env.grid.get(1, 4) is None

    env.do("s1", 0)

    obs = env.extract_state()

    assert get_var(obs, "s1").item() == 0.0
    assert env.carrying is None
    assert positions_of(env, Key) == []

def test_validation(env):
    env.reset(seed=DEFAULT_LAYOUT_SEED)

    with pytest.raises(ValueError):
        env.do("s0", (0, 0, 0))

    env.do("s0", (3, 2, 0))

    np.testing.assert_array_equal(
        get_var(env.extract_state(), "s0"),
        np.array([3.0, 2.0, 0.0], dtype=np.float32),
    )

    original_pos = tuple(env.agent_pos)
    original_goals = positions_of(env, Goal)

    with pytest.raises(ValueError):
        env.do("s3", (3, 2))

    with pytest.raises(ValueError):
        env.do("s3", (0, 0))

    with pytest.raises(ValueError):
        env.do("s2", 2)

    with pytest.raises(ValueError):
        env.do("s7", 0)

    assert tuple(env.agent_pos) == original_pos
    assert positions_of(env, Goal) == original_goals

    env.do("s1", 1)
    env.do("s2", 0, hold=True)

    env.reset(seed=DEFAULT_LAYOUT_SEED)

    assert env._clamps == {}

def test_goal_move(env):
    env.reset(seed=DEFAULT_LAYOUT_SEED)

    env.do("s3", (4, 1))

    assert positions_of(env, Goal) == [(4, 1)]
    assert env._goal_pos == (4, 1)

    np.testing.assert_array_equal(
        get_var(env.extract_state(), "s3"),
        np.array([4.0, 1.0], dtype=np.float32),
    )

def test_intervention_domain(env):
    env.reset(seed=DEFAULT_LAYOUT_SEED)

    assert len(env.intervention_domain("s0")) == 52
    assert len(env.intervention_domain("s3")) == 4

    assert env.intervention_domain("s1") == (0, 1)
    assert env.intervention_domain("s2") == (0, 1)
    assert env.intervention_domain("s5") == (0, 1)

    assert env.intervention_domain("s4") is None

def test_s5_tracks_s2_at_rho_1():
    env = FactoredGridWorld(rho=1.0)
    A = env.actions

    env.reset(seed=DEFAULT_LAYOUT_SEED)
    env.do("s0", (2, 2, 0))
    env.do("s3", (4, 4))

    s2_seen = set()

    obs, *_ = env.step(A.toggle)
    s2_seen.add(get_var(obs, "s2").item())
    assert get_var(obs, "s5").item() == get_var(obs, "s2").item()

    env.do("s1", 1)

    for action in (
        A.toggle,
        A.forward,
        A.forward,
        A.right,
        A.forward,
        A.forward,
    ):
        obs, *_ = env.step(action)
        s2 = get_var(obs, "s2").item()
        s5 = get_var(obs, "s5").item()

        s2_seen.add(s2)
        assert s5 == s2

    assert s2_seen == {0.0, 1.0}

def test_s5_inverts_s2_at_rho_0():
    env = FactoredGridWorld(rho=0.0)
    A = env.actions

    env.reset(seed=DEFAULT_LAYOUT_SEED)
    env.do("s0", (2, 2, 0))
    env.do("s3", (4, 4))

    s2_seen = set()

    obs, *_ = env.step(A.toggle)
    s2_seen.add(get_var(obs, "s2").item())
    assert get_var(obs, "s5").item() == 1.0 - get_var(obs, "s2").item()

    env.do("s1", 1)

    for action in (
        A.toggle,
        A.forward,
        A.forward,
        A.right,
        A.forward,
        A.forward,
    ):
        obs, *_ = env.step(action)
        s2 = get_var(obs, "s2").item()
        s5 = get_var(obs, "s5").item()

        s2_seen.add(s2)
        assert s5 == 1.0 - s2

    assert s2_seen == {0.0, 1.0}

def test_s5_hold_clamp():
    env = FactoredGridWorld(rho=1.0)
    A = env.actions

    env.reset(seed=DEFAULT_LAYOUT_SEED)
    env.do("s5", 1, hold=True)

    for _ in range(10):
        obs, *_ = env.step(A.forward)
        assert get_var(obs, "s5").item() == 1.0
        assert get_var(obs, "s2").item() == 0.0

def test_layout_seed_fixed():
    starts = set()
    goals = set()

    for seed in range(50):
        env = FactoredGridWorld(layout_seed=2)
        env.reset(seed=seed)

        starts.add(tuple(env.agent_pos))
        goals.add(env._goal_pos)

        assert env._door_pos == (3, 2)
        assert positions_of(env, Key) == [(1, 4)]

        assert env.agent_pos[0] < env._door_pos[0]
        assert env._goal_pos[0] == 4

    assert len(starts) > 1
    assert len(goals) > 1

def test_layout_seed_changes_layout():
    env = FactoredGridWorld(layout_seed=9)
    env.reset(seed=0)

    assert env._door_pos == (2, 2)
    assert positions_of(env, Key) == [(1, 1)]

    assert len(env.intervention_domain("s3")) == 8

def test_layout_seed_determinism():
    env1 = FactoredGridWorld(layout_seed=2)
    env2 = FactoredGridWorld(layout_seed=2)

    actions = [
        env1.actions.forward,
        env1.actions.pickup,
        env1.actions.forward,
        env1.actions.right,
        env1.actions.forward,
    ]

    obs1, _ = env1.reset(seed=11)
    obs2, _ = env2.reset(seed=11)

    trace1 = [obs1.copy()]
    trace2 = [obs2.copy()]

    for action in actions:
        obs1, *_ = env1.step(action)
        obs2, *_ = env2.step(action)

        trace1.append(obs1.copy())
        trace2.append(obs2.copy())

    assert np.array_equal(
        np.stack(trace1),
        np.stack(trace2),
    )

def test_s5_reads_clamped_s2():
    env = FactoredGridWorld(rho=1.0)
    A = env.actions

    env.reset(seed=DEFAULT_LAYOUT_SEED)

    env.do("s2", 1, hold=True)

    for _ in range(10):
        obs, *_ = env.step(A.forward)

        assert get_var(obs, "s2").item() == 1.0
        assert get_var(obs, "s5").item() == 1.0

def test_Y_zero_before_goal():
    env = FactoredGridWorld()
    obs, _ = env.reset(seed=DEFAULT_LAYOUT_SEED)
    obs, reward, terminated, truncated, info = env.step(env.actions.left)
    assert info["Y"] == 0
