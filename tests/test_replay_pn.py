
import pytest

from causalstate.oracle.replay_pn import _factored, _rollout, pn_sweep, replay_pn
from causalstate.world.envs import make_env
from causalstate.world.gridworld import FactoredGridWorld


def test_factored_returns_gridworld():
    env = make_env(rho=0.7)

    fgw = _factored(env)

    assert isinstance(fgw, FactoredGridWorld)

@pytest.mark.slow
def test_rollout_returns_expected_structure(trained_model):
    model = trained_model
    env = make_env(rho=0.7)

    start, goal, s5, actions, y = _rollout(model, env)

    assert len(start) == 3
    assert len(goal) == 2

    assert isinstance(actions, list)
    assert len(actions) > 0

    assert y in (0, 1)

@pytest.mark.slow
def test_replay_pn_expected_scores(trained_model):
    model = trained_model
    env = make_env(rho=0.7)

    assert replay_pn(model, env, "s1", n_episodes=100) == 1.0
    assert replay_pn(model, env, "s2", n_episodes=100) == 1.0
    assert replay_pn(model, env, "s4", n_episodes=100) == 0.0
    assert replay_pn(model, env, "s5", n_episodes=100) == 0.0

    s0 = replay_pn(model, env, "s0", n_episodes=100)
    assert s0 >= 0.90

    s3 = replay_pn(model, env, "s3", n_episodes=100)
    assert 0.45 <= s3 <= 0.80

    env.close()

@pytest.mark.slow
def test_replay_pn_unknown_variable(trained_model):
    model = trained_model
    env = make_env(rho=0.7)

    with pytest.raises(ValueError):
        replay_pn(model, env, "bad", n_episodes=1)

    env.close()

@pytest.mark.slow
def test_pn_sweep_returns_all_variables(trained_model):
    env = make_env(rho=0.7)

    scores = pn_sweep(
        trained_model,
        env,
        n_episodes=100,
    )

    assert set(scores.keys()) == {
        "s0","s1","s2","s3","s4","s5"
    }

    assert scores["s1"] == 1.0
    assert scores["s2"] == 1.0
    assert scores["s4"] == 0.0
    assert scores["s5"] == 0.0

@pytest.mark.slow
def test_replay_vs_reroll_contrast(weak_model):
    env = make_env(rho=0.7)

    replay = replay_pn(
        weak_model,
        env,
        "s4",
        n_episodes=80,
    )

    reroll = replay_pn(
        weak_model,
        env,
        "s4",
        n_episodes=80,
        mode="reroll",
    )

    assert replay == 0.0
    assert reroll > 0.3

    env.close()