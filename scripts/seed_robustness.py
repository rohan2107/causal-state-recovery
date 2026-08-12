import json
from pathlib import Path

from stable_baselines3 import PPO

from causalstate.agent.ppo import success_rate
from causalstate.oracle.recover import recover

SEEDS = range(5)
RHO = 0.7
N_EPISODES = 200
TAU = 0.5
BASE_SEED = 0
LAYOUT_SEED = 2
OUTPUT_PATH = Path("data/results/seed_robustness.json")


def main() -> None:
    results = []

    print("=== Seed robustness ===")

    for seed in SEEDS:
        model_path = Path(f"models/ppo_200k_rho07_seed{seed}.zip")
        model = PPO.load(model_path)

        success = success_rate(
            model,
            n_episodes=N_EPISODES,
            layout_seed=LAYOUT_SEED,
            start_seed=BASE_SEED,
            rho=RHO,
        )

        result = recover(
            model,
            n_episodes=N_EPISODES,
            tau=TAU,
            base_seed=BASE_SEED,
            layout_seed=LAYOUT_SEED,
            rho=RHO,
        )

        row = {
            "seed": seed,
            "model": str(model_path),
            "success_rate": success,
            "scores": result["scores"],
            "mask": sorted(result["mask"]),
            "recovery": result["recovery"],
            "margin": result["margin"],
            "n_episodes": N_EPISODES,
            "tau": TAU,
            "base_seed": BASE_SEED,
            "layout_seed": LAYOUT_SEED,
            "rho": RHO,
        }

        results.append(row)

        print(
            f"seed={seed} "
            f"success={success:.3f} "
            f"s0={result['scores']['s0']:.3f} "
            f"s1={result['scores']['s1']:.3f} "
            f"s2={result['scores']['s2']:.3f} "
            f"s3={result['scores']['s3']:.3f} "
            f"s4={result['scores']['s4']:.3f} "
            f"s5={result['scores']['s5']:.3f} "
            f"mask={sorted(result['mask'])} "
            f"F1={result['recovery']['f1']:.3f} "
            f"margin={result['margin']:.3f}"
        )

    artifact = {
        "config": {
            "seeds": list(SEEDS),
            "rho": RHO,
            "n_episodes": N_EPISODES,
            "tau": TAU,
            "base_seed": BASE_SEED,
            "layout_seed": LAYOUT_SEED,
        },
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w") as f:
        json.dump(artifact, f, indent=2)

    print(f"\nWrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
