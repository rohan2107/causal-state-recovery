from stable_baselines3 import PPO

from causalstate.oracle.replay_pn import (
    _counterfactual,
    _rollout,
)

PNS_VARS = ("s1", "s2", "s5")

def pns(
    model: PPO,
    env,
    var: str,
    n_episodes: int = 100,
    base_seed: int = 0,
):

    if var not in PNS_VARS:
        raise ValueError(
            f"PNS is only defined for binary variables: {PNS_VARS}"
        )

    joint = 0
    reverse_flips = 0

    on_success = 0
    off_failure = 0

    for i in range(n_episodes):

        seed = base_seed + i

        start, goal, s5, actions, _ = _rollout(
            model,
            env,
            seed=seed,
        )

        y_on = _counterfactual(
            model,
            env,
            seed,
            start,
            goal,
            s5,
            actions,
            var,
            direction="enable",
            policy=False,
            rng=None,
        )

        y_off = _counterfactual(
            model,
            env,
            seed,
            start,
            goal,
            s5,
            actions,
            var,
            direction="remove",
            policy=False,
            rng=None,
        )

        if y_on == 1:
            on_success += 1

        if y_off == 0:
            off_failure += 1

        if y_on == 1 and y_off == 0:
            joint += 1

        if y_on == 0 and y_off == 1:
            reverse_flips += 1

    return {
        "pns": joint / n_episodes,
        "reverse_flips": reverse_flips,
        "n": n_episodes,
        "p_on": on_success / n_episodes,
        "p_off": off_failure / n_episodes,
    }