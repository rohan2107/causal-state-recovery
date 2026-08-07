import numpy as np
import pytest

from causalstate.agent.ppo import ShapedReward, make_train_env, success_rate, train
from causalstate.world.gridworld import FactoredGridWorld


def test_shaped_reward_preserves_observation():
    base = FactoredGridWorld()
    shaped = ShapedReward(FactoredGridWorld())

    A = base.actions

    obs1, _ = base.reset(seed=2)
    obs2, _ = shaped.reset(seed=2)

    np.testing.assert_array_equal(obs1, obs2)
    actions = [
        A.left,
        A.forward,
        A.right,
        A.forward,
    ]

    for action in actions:
        obs1, *_ = base.step(action)
        obs2, *_ = shaped.step(action)

        np.testing.assert_array_equal(obs1, obs2)

def test_key_bonus_only_once():
    env = ShapedReward(FactoredGridWorld())
    A = env.actions

    env.reset(seed=2)
    env.do("s0", (1, 3, 1))

    # First pickup earns the shaping bonus
    _, reward, *_ = env.step(A.pickup)
    assert reward == 0.2

    # Subsequent steps should not receive it again
    _, reward, *_ = env.step(A.left)
    assert reward == 0.0

    _, reward, *_ = env.step(A.left)
    assert reward == 0.0

def test_door_bonus_only_once():
    env = ShapedReward(FactoredGridWorld())
    A = env.actions

    env.reset(seed=2)

    env.do("s0", (2, 2, 0))
    env.do("s1", 1)

    # First toggle opens the door
    _, reward, *_ = env.step(A.toggle)
    assert reward == 0.4

    # Afterwards no further shaping reward
    _, reward, *_ = env.step(A.left)
    assert reward == 0.0

    _, reward, *_ = env.step(A.left)
    assert reward == 0.0

def test_shaping_state_resets_between_episodes():
    env = ShapedReward(FactoredGridWorld())
    A = env.actions

    env.reset(seed=2)
    env.do("s0", (1, 3, 1))

    _, reward, *_ = env.step(A.pickup)
    assert reward == 0.2

    env.reset(seed=2)

    env.do("s0", (1, 3, 1))

    _, reward, *_ = env.step(A.pickup)
    assert reward == 0.2

def test_shaping_preserves_episode_outcome():
    base = FactoredGridWorld()
    shaped = ShapedReward(FactoredGridWorld())

    A = base.actions

    actions = [
        A.left,
        A.forward,
        A.right,
        A.forward,
    ]

    base.reset(seed=2)
    shaped.reset(seed=2)

    for action in actions:
        obs1, reward1, term1, trunc1, info1 = base.step(action)
        obs2, reward2, term2, trunc2, info2 = shaped.step(action)

        assert term1 == term2
        assert trunc1 == trunc2
        assert info1["Y"] == info2["Y"]

class LeftPolicy:
    def __init__(self):
        self.action = FactoredGridWorld().actions.left

    def predict(self, obs, deterministic=True):
        return self.action, None

SOLVE_123 = [
    2, 2, 2, 1,
    3,
    1,
    2, 2,
    1,
    5,
    2, 2,
    0,
    2,
]

# Optimal plan for layout_seed=2, start_seed=123.
# Re-derive by BFS if either seed changes.
class FixedPlanPolicy:
    def __init__(self, plan):
        self.plan = plan
        self.i = 0

    def predict(self, obs, deterministic=True):
        action = self.plan[self.i]
        self.i += 1
        return action, None

def test_success_rate_zero():
    model = LeftPolicy()

    assert success_rate(
        model,
        n_episodes=10,
    ) == 0.0

def test_success_rate_one():
    assert success_rate(
        FixedPlanPolicy(SOLVE_123),
        n_episodes=1,
        layout_seed=2,
        start_seed=123,
    ) == 1.0

def test_make_train_env_prunes_after_shaping():
    env = make_train_env(prune=True)

    assert env.observation_space.shape == (7,)

    obs, _ = env.reset(seed=2)

    assert obs.shape == (7,)

@pytest.mark.slow
def test_trained_model_reaches_competence(trained_model):
    assert success_rate(
        trained_model,
        n_episodes=100,
        layout_seed=2,
    ) >= 0.95

@pytest.mark.train
def test_train_reaches_competence():
    model = train(total_timesteps=200_000, seed=0, layout_seed=2)
    assert success_rate(model, n_episodes=100, layout_seed=2) >= 0.95
