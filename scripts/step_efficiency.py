import json
from pathlib import Path

from stable_baselines3 import PPO

from causalstate.world.envs import make_env

MODEL_PATH = Path("models/ppo_200k_rho07_seed0.zip")
OUTPUT_PATH = Path("data/results/step_efficiency.json")

N_EPISODES = 100
START_SEED = 123
LAYOUT_SEED = 2
MAX_STEPS = 360
RHOS = (0.7, 0.0)

def evaluate(model, rho: float) -> dict:
    env = make_env(
        layout_seed=LAYOUT_SEED,
        rho=rho,
        max_steps=MAX_STEPS,
    )

    successes = 0
    failures = 0
    success_steps = []
    failure_steps = []

    try:
        for i in range(N_EPISODES):
            obs, _ = env.reset(seed=START_SEED + i)
            steps = 0
            done = False
            success = False

            while not done:
                action, _ = model.predict(
                    obs,
                    deterministic=True,
                )

                obs, _, terminated, truncated, info = env.step(
                    int(action)
                )

                steps += 1
                done = terminated or truncated
                success = bool(info["Y"])

            if success:
                successes += 1
                success_steps.append(steps)
            else:
                failures += 1
                failure_steps.append(steps)
    finally:
        env.close()

    all_steps = success_steps + failure_steps

    return {
        "rho": rho,
        "n_episodes": N_EPISODES,
        "successes": successes,
        "failures": failures,
        "success_rate": successes / N_EPISODES,
        "mean_success_steps": (
            sum(success_steps) / len(success_steps)
            if success_steps
            else None
        ),
        "mean_failure_steps": (
            sum(failure_steps) / len(failure_steps)
            if failure_steps
            else None
        ),
        "mean_all_steps": sum(all_steps) / len(all_steps),
    }


def main() -> None:
    model = PPO.load(MODEL_PATH)

    results = [
        evaluate(model, rho)
        for rho in RHOS
    ]

    artifact = {
        "source": "deterministic_policy_rollout",
        "model": str(MODEL_PATH),
        "n_episodes": N_EPISODES,
        "start_seed": START_SEED,
        "layout_seed": LAYOUT_SEED,
        "max_steps": MAX_STEPS,
        "deterministic": True,
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w") as f:
        json.dump(artifact, f, indent=2)

    print("=== Deterministic policy step efficiency ===")

    for result in results:
        print(
            f"  rho={result['rho']:.1f} "
            f"n={result['n_episodes']} "
            f"success_rate={result['success_rate']:.3f} "
            f"success_steps={result['mean_success_steps']:.1f} "
            f"failure_steps={result['mean_failure_steps']}"
        )

    print(f"\nWrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
