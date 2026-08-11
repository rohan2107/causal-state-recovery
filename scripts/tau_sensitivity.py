import json
from pathlib import Path

from stable_baselines3 import PPO

from causalstate.evaluation.metrics import recovery_score
from causalstate.oracle.recover import recovered_mask
from causalstate.oracle.replay_pn import pn_sweep
from causalstate.world.envs import make_env

MODEL_PATH = Path("models/ppo_200k_rho07_seed0.zip")
OUTPUT_PATH = Path("data/results/tau_sensitivity.json")

TAUS = (0.1, 0.3, 0.5, 0.6, 0.7, 0.9)


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
        scores = pn_sweep(
            model,
            env,
            n_episodes=n_episodes,
            base_seed=base_seed,
        )
    finally:
        env.close()

    results = []

    for tau in TAUS:
        mask = recovered_mask(scores, tau)
        recovery = recovery_score(mask)

        results.append(
            {
                "tau": tau,
                "mask": sorted(mask),
                "recovery": recovery,
            }
        )

    artifact = {
        "model": str(MODEL_PATH),
        "n_episodes": n_episodes,
        "base_seed": base_seed,
        "layout_seed": layout_seed,
        "rho": rho,
        "scores": scores,
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w") as f:
        json.dump(artifact, f, indent=2)

    print("PN scores:")
    for var, score in scores.items():
        print(f"  {var}: {score:.3f}")

    print("\nTau sensitivity:")
    for result in results:
        print(
            f"  tau={result['tau']:.1f} "
            f"mask={result['mask']} "
            f"F1={result['recovery']['f1']:.3f}"
        )

    print(f"\nWrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
