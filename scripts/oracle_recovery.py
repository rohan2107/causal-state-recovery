import json
from pathlib import Path

from stable_baselines3 import PPO

from causalstate.oracle.recover import recover

MODEL_PATH = Path("models/ppo_200k_rho07_seed0.zip")
OUTPUT_PATH = Path("data/results/oracle_recovery.json")


def main() -> None:
    n_episodes = 200
    tau = 0.5
    base_seed = 0
    layout_seed = 2
    rho = 0.7

    model = PPO.load(MODEL_PATH)

    result = recover(
        model,
        n_episodes=n_episodes,
        tau=tau,
        base_seed=base_seed,
        layout_seed=layout_seed,
        rho=rho,
    )

    artifact = {
        "model": str(MODEL_PATH),
        "n_episodes": n_episodes,
        "tau": tau,
        "base_seed": base_seed,
        "layout_seed": layout_seed,
        "rho": rho,
        "scores": result["scores"],
        "mask": sorted(result["mask"]),
        "recovery": result["recovery"],
        "margin": result["margin"],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w") as f:
        json.dump(artifact, f, indent=2)

    print("Recovery scores:")
    for var, score in result["scores"].items():
        print(f"  {var}: {score:.3f}")

    print(f"Recovered mask: {sorted(result['mask'])}")
    print(f"Recovery: {result['recovery']}")
    print(f"Margin: {result['margin']:.3f}")
    print(f"Wrote: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
