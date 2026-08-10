import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from causalstate.observational.dataset import (
    collect_observational,
    content_hash,
)
from causalstate.world.envs import make_env

MODEL_PATH = Path("tests/.cache/ppo_200k_rho07_seed0.zip")
DATA_DIR = Path("data/observational")

EPS = 0.45
N_EPISODES = 400
SEEDS = (0, 1, 2)

START_SEEDS = {
    0: 2000,
    1: 2400,
    2: 2800,
}


def main() -> None:
    model = PPO.load(MODEL_PATH)
    summaries = []

    for seed in SEEDS:
        start_seed = START_SEEDS[seed]
        env = make_env(rho=0.7)

        dataset = collect_observational(
            model,
            env,
            n_episodes=N_EPISODES,
            eps_pool=(EPS,),
            start_seed=start_seed,
            rng=np.random.default_rng(seed),
        )

        env.close()

        output = DATA_DIR / f"baseline_summaries_seed{seed}.npz"

        np.savez_compressed(
            output,
            obs=dataset["obs"],
            Y=dataset["Y"],
            episode_id=dataset["episode_id"],
        )

        y_rate = float(
            np.array(
                [
                    dataset["Y"][
                        dataset["episode_id"] == episode
                    ][0]
                    for episode in np.unique(dataset["episode_id"])
                ]
            ).mean()
        )

        summaries.append(
            {
                "seed": seed,
                "start_seed": start_seed,
                "n_episodes": N_EPISODES,
                "eps": EPS,
                "y_rate": y_rate,
                "content_hash": content_hash(dataset),
            }
        )

        print(
            f"seed={seed} "
            f"start_seed={start_seed} "
            f"y_rate={y_rate:.4f}"
        )

    metadata = {
        "eps": EPS,
        "n_episodes": N_EPISODES,
        "seeds": list(SEEDS),
        "summaries": summaries,
    }

    metadata_path = DATA_DIR / "baseline_summaries.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
