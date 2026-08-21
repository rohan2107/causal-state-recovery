import json
from pathlib import Path

from stable_baselines3 import PPO

from causalstate.agent.ppo import train
from causalstate.world.envs import make_env, pruned_obs
from causalstate.world.gridworld import CAUSAL_ORACLE

OUTPUT_PATH = Path("data/results/step_efficiency_matrix.json")

SEEDS = range(5)
TRAIN_RHO = 0.7
OOD_RHO = 0.0
N_EPISODES = 100
LAYOUT_SEED = 2
START_SEED = 123
TOTAL_TIMESTEPS = 200_000

EXPECTED_SUCCESS = {
    "full": {
        0: (1.00, 1.00),
        1: (1.00, 1.00),
        2: (0.95, 0.79),
        3: (1.00, 0.92),
        4: (1.00, 1.00),
    },
    "pruned": {
        0: (1.00, 1.00),
        1: (1.00, 1.00),
        2: (1.00, 1.00),
        3: (1.00, 1.00),
        4: (1.00, 1.00),
    },
}


def evaluate(model, rho, prune):
    env = make_env(
        layout_seed=LAYOUT_SEED,
        rho=rho,
    )

    if prune:
        env = pruned_obs(env, CAUSAL_ORACLE)

    per_episode = []

    for i in range(N_EPISODES):
        obs, _ = env.reset(seed=START_SEED + i)
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1

        per_episode.append({
            "episode": i,
            "steps": steps,
            "success": bool(info.get("Y", 0)),
        })

    env.close()

    success_steps = [
        x["steps"] for x in per_episode if x["success"]
    ]
    failure_steps = [
        x["steps"] for x in per_episode if not x["success"]
    ]

    return {
        "success_rate": len(success_steps) / N_EPISODES,
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
        "success_steps": success_steps,
        "failure_steps": failure_steps,
        "per_episode": per_episode,
    }


def check_expected(condition, seed, train_result, ood_result):
    expected_train, expected_ood = EXPECTED_SUCCESS[condition][seed]

    train_ok = train_result["success_rate"] == expected_train
    ood_ok = ood_result["success_rate"] == expected_ood

    status = "OK" if train_ok and ood_ok else "MISMATCH"

    print(
        f"  reproduction: {status} "
        f"(train {train_result['success_rate']:.2f}, "
        f"OOD {ood_result['success_rate']:.2f})"
    )

    return {
        "train": train_ok,
        "ood": ood_ok,
        "ok": train_ok and ood_ok,
    }


def main():
    results = {
        "full": [],
        "pruned": [],
    }

    for condition in ("full", "pruned"):
        prune = condition == "pruned"

        for seed in SEEDS:
            print(f"=== {condition} seed={seed} ===")

            if prune:
                model_path = Path(
                    f"models/ppo_200k_rho07_seed{seed}_pruned.zip"
                )

                model = train(
                    total_timesteps=TOTAL_TIMESTEPS,
                    seed=seed,
                    layout_seed=LAYOUT_SEED,
                    rho=TRAIN_RHO,
                    prune=True,
                    save_path=model_path,
                )
            else:
                model_path = Path(
                    f"models/ppo_200k_rho07_seed{seed}.zip"
                )
                model = PPO.load(model_path)

            train_result = evaluate(
                model,
                TRAIN_RHO,
                prune,
            )
            ood_result = evaluate(
                model,
                OOD_RHO,
                prune,
            )

            reproduction = check_expected(
                condition,
                seed,
                train_result,
                ood_result,
            )

            results[condition].append({
                "seed": seed,
                "train": train_result,
                "ood": ood_result,
                "reproduction": reproduction,
            })

    artifact = {
        "config": {
            "seeds": list(SEEDS),
            "total_timesteps": TOTAL_TIMESTEPS,
            "train_rho": TRAIN_RHO,
            "ood_rho": OOD_RHO,
            "n_episodes": N_EPISODES,
            "layout_seed": LAYOUT_SEED,
            "start_seed": START_SEED,
        },
        "results": results,
        "reproduction_ok": all(
            r["reproduction"]["ok"]
            for condition in results.values()
            for r in condition
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w") as f:
        json.dump(artifact, f, indent=2)

    print(f"\nreproduction_ok: {artifact['reproduction_ok']}")
    print(f"saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()