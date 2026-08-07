import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box

from causalstate.world.gridworld import (
    ALL_VARS,
    OBS_HIGH,
    OBS_LOW,
    VAR_SLICES,
    FactoredGridWorld,
)


def make_env(
    start_seed=None,
    layout_seed=2,
    rho=None,
    noise_std=1.0,
    max_steps=360,
    **kwargs,
):
    env = FactoredGridWorld(
        rho=rho,
        layout_seed=layout_seed,
        noise_std=noise_std,
        max_steps=max_steps,
        **kwargs,
    )

    if start_seed is not None:
        env.reset(seed=start_seed)

    return env

class PrunedObservationWrapper(ObservationWrapper):
    def __init__(self, env, keep):
        super().__init__(env)
        self.keep = tuple(keep)
        unknown = set(self.keep) - set(ALL_VARS)
        if unknown:
            raise ValueError(f"Unknown variables: {sorted(unknown)}")
        indices = []
        for var in ALL_VARS:
            if var in self.keep:
                indices.extend(
                    range(*VAR_SLICES[var].indices(len(OBS_LOW)))
                )

        self.indices = np.array(indices, dtype=int)
        self.observation_space = Box(
            low=OBS_LOW[self.indices],
            high=OBS_HIGH[self.indices],
            dtype=np.float32,
        )

    def observation(self, obs):
        return obs[self.indices]

    def do(self, *args, **kwargs):
        return self.env.do(*args, **kwargs)

    def release(self, *args, **kwargs):
        return self.env.release(*args, **kwargs)

    @property
    def actions(self):
        return self.env.actions

    def intervention_domain(self, *args, **kwargs):
        return self.env.intervention_domain(*args, **kwargs)

    def extract_state(self):
        return self.env.extract_state()

def pruned_obs(env, keep):
    return PrunedObservationWrapper(env, keep)
