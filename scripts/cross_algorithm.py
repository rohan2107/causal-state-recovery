import json
from pathlib import Path

from sb3_contrib import TRPO
from stable_baselines3 import PPO

from causalstate.agent.ppo import success_rate, train
from causalstate.oracle.recover import recover

OUTPUT_PATH = Path("data/results/cross_algorithm.json")

TOTAL_TIMESTEPS = 200_000
SEED = 0
LAYOUT_SEED = 2
RHO = 0.7
N_EPISODES = 200
COMPETENCE_THRESHOLD = 0.5


def evaluate(model, algorithm: str) -> dict:
    success = success_rate(
        model,
        n_episodes=100,
        layout_seed=LAYOUT_SEED,
        rho=RHO,
    )

    result = {
        "algorithm": algorithm,
        "success_rate": success,
        "competent": success > COMPETENCE_THRESHOLD,
    }

    if result["competent"]:
        recovery = recover(
            model,
            n_episodes=N_EPISODES,
            layout_seed=LAYOUT_SEED,
            rho=RHO,
        )
        result["scores"] = recovery["scores"]
        result["mask"] = sorted(recovery["mask"])
        result["recovery"] = recovery["recovery"]
        result["margin"] = recovery["margin"]

    return result


def main() -> None:
    results = []

    print("=== PPO ===")
    ppo = train(
        total_timesteps=TOTAL_TIMESTEPS,
        seed=SEED,
        layout_seed=LAYOUT_SEED,
        rho=RHO,
        prune=False,
        ent_coef=0.05,
        algo=PPO,
    )
    ppo_result = evaluate(ppo, "PPO")
    results.append(ppo_result)

    print(f"success={ppo_result['success_rate']:.3f}")
    print(f"competent={ppo_result['competent']}")

    print("\n=== TRPO ===")
    trpo = train(
        total_timesteps=TOTAL_TIMESTEPS,
        seed=SEED,
        layout_seed=LAYOUT_SEED,
        rho=RHO,
        prune=False,
        ent_coef=None,
        algo=TRPO,
    )
    trpo_result = evaluate(trpo, "TRPO")
    results.append(trpo_result)

    print(f"success={trpo_result['success_rate']:.3f}")
    print(f"competent={trpo_result['competent']}")

    if trpo_result["competent"]:
        print("PN:")
        for var, score in trpo_result["scores"].items():
            print(f"  {var}: {score:.3f}")
        print(f"mask={trpo_result['mask']}")
        print(f"F1={trpo_result['recovery']['f1']:.3f}")

    artifact = {
        "config": {
            "total_timesteps": TOTAL_TIMESTEPS,
            "seed": SEED,
            "layout_seed": LAYOUT_SEED,
            "rho": RHO,
            "n_episodes": N_EPISODES,
            "competence_threshold": COMPETENCE_THRESHOLD,
        },
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w") as f:
        json.dump(artifact, f, indent=2)

    print(f"\nWrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
