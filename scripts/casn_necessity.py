import json
from pathlib import Path

from stable_baselines3 import PPO

from causalstate.evaluation.report import report
from causalstate.observational.casn import (
    intervention_magnitude,
    train_casn,
)
from causalstate.observational.dataset import (
    collect_observational,
    split_by_episode,
)
from causalstate.observational.mlp import make_dataloader
from causalstate.world.envs import make_env

MODEL_PATH = Path("models/ppo_200k_rho07_seed0.zip")
OUTPUT_PATH = Path("data/results/ablation_casn_necessity.json")

RHOS = (0.7, 1.0)
TARGET_LAMBDAS = (0.0, 0.1, 0.5, 1.0)
SEEDS = (0, 1, 2)

N_EPISODES = 600
START_SEED = 777
SPLIT_SEED = 0
LAYOUT_SEED = 2

EPOCHS = 40
LR = 1e-3
HIDDEN = 64
REP_DIM = 32
BATCH_SIZE = 256
L1 = 0.02
INT_LAMBDA = 0.001
INT_REG = 0.01
BIAS = 1.0
THRESHOLD = 0.1


def fit_one(
    splits: dict,
    *,
    target_lambda: float,
    seed: int,
) -> dict:
    model, scaler = train_casn(
        splits["train"],
        epochs=EPOCHS,
        lr=LR,
        hidden=HIDDEN,
        rep_dim=REP_DIM,
        batch_size=BATCH_SIZE,
        l1=L1,
        target_lambda=target_lambda,
        int_lambda=INT_LAMBDA,
        int_reg=INT_REG,
        bias=BIAS,
        seed=seed,
    )

    scores = model.gate_values()

    result = report(
        scores,
        model,
        scaler,
        splits,
        threshold=THRESHOLD,
    )

    train_loader, _ = make_dataloader(
        splits["train"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=seed,
    )

    result["intervention_magnitude"] = intervention_magnitude(
        model,
        train_loader,
    )

    return result


def main() -> None:
    model = PPO.load(MODEL_PATH)

    results = []

    for rho in RHOS:
        print(f"=== rho={rho:.1f} ===")

        env = make_env(
            layout_seed=LAYOUT_SEED,
            rho=rho,
        )

        try:
            dataset = collect_observational(
                model,
                env,
                n_episodes=N_EPISODES,
                start_seed=START_SEED,
            )
        finally:
            env.close()

        splits = split_by_episode(
            dataset,
            seed=SPLIT_SEED,
        )

        y_rate = float(dataset["Y"].mean())

        print(
            f"episodes={N_EPISODES} "
            f"rows={len(dataset['Y'])} "
            f"Y-rate={y_rate:.3f}"
        )

        for target_lambda in TARGET_LAMBDAS:
            for seed in SEEDS:
                print(
                    f"  target_lambda={target_lambda:.1f} "
                    f"seed={seed}"
                )

                result = fit_one(
                    splits,
                    target_lambda=target_lambda,
                    seed=seed,
                )

                artifact_result = {
                    "rho": rho,
                    "target_lambda": target_lambda,
                    "seed": seed,
                    "n_episodes": N_EPISODES,
                    "n_rows": len(dataset["Y"]),
                    "y_rate": y_rate,
                    "scores": result["scores"],
                    "mask": result["mask"],
                    "separation": result["separation"],
                    "recovery": result["recovery"],
                    "accuracies": result["accuracies"],
                    "intervention_magnitude": (
                        result["intervention_magnitude"]
                    ),
                }

                results.append(artifact_result)

                print(
                    f"    F1={result['recovery']['f1']:.3f} "
                    f"mask={result['mask']} "
                    f"Delta={result['intervention_magnitude']:.3f}"
                )

        print()

    artifact = {
        "config": {
            "model": str(MODEL_PATH),
            "rhos": list(RHOS),
            "target_lambdas": list(TARGET_LAMBDAS),
            "seeds": list(SEEDS),
            "n_episodes": N_EPISODES,
            "start_seed": START_SEED,
            "split_seed": SPLIT_SEED,
            "layout_seed": LAYOUT_SEED,
            "epochs": EPOCHS,
            "lr": LR,
            "hidden": HIDDEN,
            "rep_dim": REP_DIM,
            "batch_size": BATCH_SIZE,
            "l1": L1,
            "int_lambda": INT_LAMBDA,
            "int_reg": INT_REG,
            "bias": BIAS,
            "threshold": THRESHOLD,
            "necessity_off_definition": (
                "target_lambda=0; int_lambda unchanged"
            ),
        },
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(artifact, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
