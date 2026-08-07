import pytest

from causalstate.oracle.pns import pns
from causalstate.world.envs import make_env


@pytest.mark.slow
def test_pns_expected_scores(trained_model):

    env = make_env(rho=0.7)

    assert pns(
        trained_model,
        env,
        "s1",
        n_episodes=100,
    )["pns"] == 1.0

    assert pns(
        trained_model,
        env,
        "s2",
        n_episodes=100,
    )["pns"] == 1.0

    assert pns(
        trained_model,
        env,
        "s5",
        n_episodes=100,
    )["pns"] == 0.0

    env.close()

@pytest.mark.slow
def test_reverse_flips_are_zero(trained_model):

    env = make_env(rho=0.7)

    for var in ("s1", "s2", "s5"):

        result = pns(
            trained_model,
            env,
            var,
            n_episodes=100,
        )

        assert result["reverse_flips"] == 0

    env.close()

def test_pns_rejects_non_binary_variable():

    env = make_env(rho=0.7)

    with pytest.raises(ValueError):
        pns(
            None,
            env,
            "s4",
            n_episodes=1,
        )