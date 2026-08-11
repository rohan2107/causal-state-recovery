import json
from pathlib import Path

from stable_baselines3 import PPO

from causalstate.agent.ppo import success_rate
from causalstate.oracle.replay_pn import replay_pn
from causalstate.world.envs import make_env

OUTPUT_PATH = Path("data/results/replay_vs_reroll.json")

BUDGETS = (100_000, 150_000, 200_000)
N_EPISODES = 200
BASE_SEED = 0
LAYOUT_SEED = 2
RHO = 0.7
VARIABLE = "s4"


def main() -> None:
    results = []

    for timesteps in BUDGETS:
        model_path = Path(
            f"models/ppo_{timesteps // 1000}k_rho07_seed0.zip"
        )

        print(f"=== {timesteps // 1000}k ===")
        print(f"model: {model_path}")

        model = PPO.load(model_path)

        success = success_rate(
            model,
            n_episodes=N_EPISODES,
            layout_seed=LAYOUT_SEED,
            start_seed=BASE_SEED,
            rho=RHO,
        )

        env = make_env(
            layout_seed=LAYOUT_SEED,
            rho=RHO,
        )

        try:
            replay = replay_pn(
                model,
                env,
                VARIABLE,
                n_episodes=N_EPISODES,
                mode="replay",
                base_seed=BASE_SEED,
            )

            reroll = replay_pn(
                model,
                env,
                VARIABLE,
                n_episodes=N_EPISODES,
                mode="reroll",
                base_seed=BASE_SEED,
            )
        finally:
            env.close()

        result = {
            "timesteps": timesteps,
            "model": str(model_path),
            "success_rate": success,
            "replay_pn": replay,
            "reroll_pn": reroll,
        }

        results.append(result)

        print(f"success={success:.3f}")
        print(f"replay-PN({VARIABLE})={replay:.3f}")
        print(f"reroll-PN({VARIABLE})={reroll:.3f}")
        print()

    artifact = {
        "config": {
            "budgets": list(BUDGETS),
            "n_episodes": N_EPISODES,
            "base_seed": BASE_SEED,
            "layout_seed": LAYOUT_SEED,
            "rho": RHO,
            "variable": VARIABLE,
        },
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w") as f:
        json.dump(artifact, f, indent=2)

    print(f"Wrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
