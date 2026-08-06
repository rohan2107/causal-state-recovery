import numpy as np
from minigrid.core.world_object import Wall
from stable_baselines3 import PPO

from causalstate.world.gridworld import (
    ALL_VARS,
    FactoredGridWorld,
    get_var,
)

RECOVERY_VARS = ALL_VARS

def _factored(env):
    while not isinstance(env, FactoredGridWorld):
        env = env.env
    return env

def _rollout(model: PPO, env, seed=None):
    if seed is None:
        obs, _ = env.reset()
    else:
        obs, _ = env.reset(seed=seed)
    start = tuple(int(x) for x in get_var(obs, "s0").tolist())
    goal = tuple(int(x) for x in get_var(obs, "s3").tolist())
    actions, info, done = [], {}, False
    spurious = int(get_var(obs, "s5").item())
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        actions.append(int(a))
        obs, _r, term, trunc, info = env.step(int(a))
        done = term or trunc
    return start, goal, spurious, actions, int(info.get("Y", 0))

def _room_cells(board, cols, rows):
    return [
        (i, j)
        for i in cols
        for j in rows
        if not isinstance(board.grid.get(i, j), Wall)
    ]

def _other(cells, current, rng):
    choices = [c for c in cells if c != tuple(current)]
    return choices[int(rng.integers(len(choices)))] if choices else tuple(current)

def replay_pn(
    model: PPO,
    env,
    var: str,
    n_episodes: int = 100,
    rng=None,
    mode="replay",
    base_seed=0,
):
    rng = rng if rng is not None else np.random.default_rng(0)

    fgw = _factored(env)

    if var not in RECOVERY_VARS:
        raise ValueError(f"Unknown recovery variable: {var}")

    if mode not in ("replay", "reroll"):
        raise ValueError(f"Unknown mode: {mode}")

    flips = 0
    considered = 0

    for i in range(n_episodes):
        seed = base_seed + i
        start, goal, s5, actions, y = _rollout(model, env, seed=seed)

        if y != 1:
            continue

        considered += 1
        env.reset(seed=seed)

        b = env.unwrapped
        door_col = fgw._door_pos[0]

        if var == "s0":
            left = _room_cells(
                b,
                range(1, door_col),
                range(1, b.height - 1),
            )
            col, row = _other(left, start[:2], rng)
            fgw.do("s0", (col, row, int(rng.integers(4))))
        else:
            fgw.do("s0", start)

        if var == "s3":
            right = _room_cells(
                b,
                range(door_col + 1, b.width - 1),
                range(1, b.height - 1),
            )
            fgw.do("s3", _other(right, goal, rng))
        else:
            fgw.do("s3", goal)

        info = {}

        if var == "s4":
            fgw.do(
                "s4",
                float(rng.standard_normal()),
                hold=True,
            )

        if var == "s5":
            fgw.do(
                "s5",
                1 - s5,
                hold=True,
            )

        if mode == "replay":

            for action in actions:
                if var == "s1":
                    fgw.do("s1", 0)
                elif var == "s2":
                    fgw.do("s2", 0)

                _, _, terminated, truncated, info = env.step(action)

                if terminated or truncated:
                    break

        else:
            obs = fgw.gen_obs()
            done = False

            while not done:
                if var == "s1":
                    fgw.do("s1", 0)
                elif var == "s2":
                    fgw.do("s2", 0)

                action, _ = model.predict(
                    obs,
                    deterministic=True,
                )

                obs, _, terminated, truncated, info = env.step(int(action))

                done = terminated or truncated

        if info.get("Y", 0) == 0:
            flips += 1

    if considered == 0:
        raise RuntimeError("Replay-PN found no successful episodes.")

    return flips / considered

def pn_sweep(
    model,
    env,
    variables=RECOVERY_VARS,
    n_episodes=100,
    rng=None,
    base_seed=0,
):
    scores = {}

    for var in variables:
        scores[var] = replay_pn(
            model,
            env,
            var,
            n_episodes=n_episodes,
            rng=rng,
            base_seed=base_seed,
        )

    return scores