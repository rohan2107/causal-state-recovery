from pathlib import Path

import pytest
from stable_baselines3 import PPO

from causalstate.agent.ppo import train

MODEL_PATH = Path("tests/.cache/ppo_200k_rho07_seed0.zip")
MODEL_PATH_100K = Path("tests/.cache/ppo_100k_rho07_seed0.zip")

@pytest.fixture(scope="session")
def trained_model():
    if MODEL_PATH.exists():
        return PPO.load(MODEL_PATH)

    model = train(total_timesteps=200_000, seed=0, rho=0.7,)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    return model

@pytest.fixture(scope="session")
def weak_model():
    if MODEL_PATH_100K.exists():
        return PPO.load(MODEL_PATH_100K)

    model = train(
        total_timesteps=100_000,
        seed=0,
        rho=0.7,
    )

    MODEL_PATH_100K.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH_100K)
    return model