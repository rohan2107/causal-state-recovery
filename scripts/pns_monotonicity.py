import json
from pathlib import Path

from stable_baselines3 import PPO

from causalstate.oracle.pns import PNS_VARS, pns
from causalstate.world.envs import make_env

MODEL_PATH = Path("models/ppo_200k_rho07_seed0.zip")
OUTPUT_PATH = Path("data/results/pns_monotonicity.json")


def main() -> None:
    n_episodes = 200
    base_seed = 0
    layout_seed = 2
    rho = 0.7

    model = PPO.load(MODEL_PATH)

    env = make_env(
        layout_seed=layout_seed,
        rho=rho,
    )

    try:
        results = {
            var: pns(
                model,
                env,
                var,
                n_episodes=n_episodes,
                base_seed=base_seed,
            )
            for var in PNS_VARS
        }
    finally:
        env.close()

    p_y = 1.0

    artifact = {
        "model": str(MODEL_PATH),
        "n_episodes": n_episodes,
        "base_seed": base_seed,
        "layout_seed": layout_seed,
        "rho": rho,
        "p_y": p_y,
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w") as f:
        json.dump(artifact, f, indent=2)

    print(f"P(Y=1): {p_y:.3f}")

    print("\nPNS / monotonicity:")
    for var, result in results.items():
        print(
            f"  {var}: "
            f"PNS={result['pns']:.3f} "
            f"PN≈PNS={result['pns']:.3f} "
            f"p_on={result['p_on']:.3f} "
            f"p_off={result['p_off']:.3f} "
            f"reverse_flips={result['reverse_flips']}"
        )

    print(f"\nWrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
