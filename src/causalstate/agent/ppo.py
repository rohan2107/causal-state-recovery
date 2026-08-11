from __future__ import annotations

import copy
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

from causalstate.world.envs import make_env, pruned_obs
from causalstate.world.gridworld import CAUSAL_ORACLE, get_var


class ShapedReward(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        key_bonus: float = 0.2,
        door_bonus: float = 0.2,
    ):
        super().__init__(env)

        self._key_bonus = key_bonus
        self._door_bonus = door_bonus

        self._got_key = False
        self._opened = False

    @property
    def actions(self):
        return self.env.actions

    @property
    def intervention_domain(self):
        return self.env.intervention_domain

    def extract_state(self):
        return self.env.extract_state()

    def do(self, *args, **kwargs):
        return self.env.do(*args, **kwargs)

    def release(self, *args, **kwargs):
        return self.env.release(*args, **kwargs)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self._got_key = False
        self._opened = False

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if (
            not self._got_key
            and get_var(obs, "s1").item() > 0.5
        ):
            reward += self._key_bonus
            self._got_key = True

        if (
            not self._opened
            and get_var(obs, "s2").item() > 0.5
        ):
            reward += self._door_bonus
            self._opened = True

        return obs, reward, terminated, truncated, info

def make_train_env(
    layout_seed=2,
    rho=None,
    prune=False,
):
    env = make_env(
        layout_seed=layout_seed,
        rho=rho,
    )

    env = ShapedReward(env)

    if prune:
        env = pruned_obs(env, CAUSAL_ORACLE)

    return env

def success_rate(
    model,
    n_episodes=100,
    layout_seed=2,
    start_seed=123,
    rho=None,
    prune=False,
):
    env = make_env(
        layout_seed=layout_seed,
        rho=rho,
    )
    if prune:
        env = pruned_obs(env, CAUSAL_ORACLE)

    successes = 0
    for i in range(n_episodes):
        obs, _ = env.reset(seed=start_seed + i)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

        if info.get("Y", 0) == 1:
            successes += 1

    env.close()
    return successes / n_episodes

def train(
    total_timesteps=200_000,
    seed=0,
    layout_seed=2,
    rho=None,
    prune=False,
    save_path=None,
    ent_coef=0.05,
    eval_every=25_000,
    eval_episodes=50,
    algo=PPO,
    **algo_kwargs,
):
    env = make_train_env(
        layout_seed=layout_seed,
        rho=rho,
        prune=prune,
    )

    kwargs = dict(algo_kwargs)

    if ent_coef is not None:
        kwargs["ent_coef"] = ent_coef

    model = algo(
        "MlpPolicy",
        env,
        seed=seed,
        verbose=0,
        **kwargs,
    )

    best_rate = -1.0
    best_params = None
    remaining = total_timesteps
    while remaining > 0:
        chunk = min(eval_every, remaining)
        model.learn(
            total_timesteps=chunk,
            reset_num_timesteps=False,
        )
        remaining -= chunk
        rate = success_rate(
            model,
            n_episodes=eval_episodes,
            layout_seed=layout_seed,
            rho=rho,
            prune=prune,
        )

        if rate >= best_rate:
            best_rate = rate
            best_params = copy.deepcopy(model.get_parameters())

    if best_params is not None:
        model.set_parameters(best_params)

    if save_path is not None:
        Path(save_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        model.save(save_path)

    return model
