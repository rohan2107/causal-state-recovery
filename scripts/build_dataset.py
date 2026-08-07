import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from causalstate.agent.ppo import success_rate
from causalstate.observational.dataset import (
    EPS_POOL,
    collect_observational,
    content_hash,
    save_split,
    split_by_episode,
)
from causalstate.world.envs import make_env

MODEL_PATH = Path("tests/.cache/ppo_200k_rho07_seed0.zip")

OUTPUT_DIR = Path("data/observational")

N_EPISODES = 600

COLLECT_START_SEED = 777

def y_rate(split: dict[str, np.ndarray]) -> float:
    episode_ids = np.unique(split["episode_id"])

    y = np.array([
        split["Y"][split["episode_id"] == e][0]
        for e in episode_ids
    ])

    return float(y.mean())

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = PPO.load(MODEL_PATH)
    policy_success = success_rate(
        model,
        n_episodes=100,
    )
    train_env = make_env(rho=0.7)

    dataset = collect_observational(
        model,
        train_env,
        n_episodes=N_EPISODES,
        start_seed=COLLECT_START_SEED,
    )

    train_env.close()

    splits = split_by_episode(dataset)

    for name, split in splits.items():

        save_split(
            OUTPUT_DIR / f"{name}.npz",
            split,
        )
    ood_env = make_env(rho=0.0)

    ood = collect_observational(
        model,
        ood_env,
        n_episodes=N_EPISODES,
        start_seed=COLLECT_START_SEED + N_EPISODES,
    )

    ood_env.close()

    save_split(
        OUTPUT_DIR / "ood_test.npz",
        ood,
    )

    metadata = {
        "n_episodes": N_EPISODES,
        "collect_start_seed": COLLECT_START_SEED,
        "eps_pool": list(EPS_POOL),
        "train_rho": 0.7,
        "ood_rho": 0.0,

        "policy_success_rate": policy_success,
        "observational_y_rate": y_rate(dataset),
        "train_y_rate": y_rate(splits["train"]),
        "val_y_rate": y_rate(splits["val"]),
        "test_y_rate": y_rate(splits["test"]),
        "ood_test_y_rate": y_rate(ood),

        "train_hash": content_hash(splits["train"]),
        "val_hash": content_hash(splits["val"]),
        "test_hash": content_hash(splits["test"]),
        "ood_test_hash": content_hash(ood),
    }

    with open(
        OUTPUT_DIR / "metadata.json",
        "w",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print(f"Policy success : {metadata['policy_success_rate']:.3f}")
    print(f"Observed Y-rate: {metadata['observational_y_rate']:.3f}")
    print(f"Train Y-rate   : {metadata['train_y_rate']:.3f}")
    print(f"Val Y-rate     : {metadata['val_y_rate']:.3f}")
    print(f"Test Y-rate    : {metadata['test_y_rate']:.3f}")
    print(f"OOD Y-rate     : {metadata['ood_test_y_rate']:.3f}")
    print()
    print(f"Dataset written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
