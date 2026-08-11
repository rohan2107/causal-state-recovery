import json
from pathlib import Path

from causalstate.agent.ppo import success_rate, train

OUTPUT_PATH = Path("data/results/downstream_rq2.json")

SEEDS = range(5)
TRAIN_RHO = 0.7
OOD_RHO = 0.0
N_EPISODES = 100
COMPETENCE_THRESHOLD = 0.5
TOTAL_TIMESTEPS = 200_000
LAYOUT_SEED = 2


def evaluate_policy(model, prune: bool) -> dict:
    train_success = success_rate(
        model,
        n_episodes=N_EPISODES,
        layout_seed=LAYOUT_SEED,
        rho=TRAIN_RHO,
        prune=prune,
    )

    ood_success = success_rate(
        model,
        n_episodes=N_EPISODES,
        layout_seed=LAYOUT_SEED,
        rho=OOD_RHO,
        prune=prune,
    )

    competent = train_success > COMPETENCE_THRESHOLD

    return {
        "train_success": train_success,
        "ood_success": ood_success,
        "competent": competent,
        "ood_gap": (
            ood_success - train_success
            if competent
            else None
        ),
    }


def summarise(results: list[dict]) -> dict:
    competent = [
        result
        for result in results
        if result["competent"]
    ]

    return {
        "n_total": len(results),
        "n_competent": len(competent),
        "train_mean": (
            sum(r["train_success"] for r in competent)
            / len(competent)
            if competent
            else None
        ),
        "ood_mean": (
            sum(r["ood_success"] for r in competent)
            / len(competent)
            if competent
            else None
        ),
        "ood_gap_mean": (
            sum(r["ood_gap"] for r in competent)
            / len(competent)
            if competent
            else None
        ),
    }


def main() -> None:
    all_results = {
        "full": [],
        "pruned": [],
    }

    for prune in (False, True):
        condition = "pruned" if prune else "full"

        print(f"=== {condition.upper()} ===")

        for seed in SEEDS:
            print(f"seed={seed}: training...", flush=True)

            model = train(
                total_timesteps=TOTAL_TIMESTEPS,
                seed=seed,
                layout_seed=LAYOUT_SEED,
                rho=TRAIN_RHO,
                prune=prune,
            )

            result = evaluate_policy(model, prune)
            result["seed"] = seed
            result["condition"] = condition

            all_results[condition].append(result)

            status = (
                "competent"
                if result["competent"]
                else "FAILED TRAINING"
            )

            print(
                f"  train={result['train_success']:.2f} "
                f"OOD={result['ood_success']:.2f} "
                f"{status}"
            )

    artifact = {
        "config": {
            "seeds": list(SEEDS),
            "total_timesteps": TOTAL_TIMESTEPS,
            "train_rho": TRAIN_RHO,
            "ood_rho": OOD_RHO,
            "n_episodes": N_EPISODES,
            "competence_threshold": COMPETENCE_THRESHOLD,
            "layout_seed": LAYOUT_SEED,
        },
        "results": all_results,
        "summary": {
            "full": summarise(all_results["full"]),
            "pruned": summarise(all_results["pruned"]),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w") as f:
        json.dump(artifact, f, indent=2)

    print("\n=== Competent-conditional summary ===")

    for condition in ("full", "pruned"):
        summary = artifact["summary"][condition]
        print(
            f"{condition}: "
            f"{summary['n_competent']}/{summary['n_total']} competent, "
            f"train={summary['train_mean']:.2f}, "
            f"OOD={summary['ood_mean']:.2f}, "
            f"gap={summary['ood_gap_mean']:+.2f}"
        )

    print(f"\nWrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
