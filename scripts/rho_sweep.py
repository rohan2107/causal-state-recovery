import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from causalstate.observational.casn import run_casn
from causalstate.observational.dataset import (
    collect_observational,
    split_by_episode,
)
from causalstate.observational.sparse_gate import run_sparse_gate
from causalstate.world.envs import make_env
from causalstate.world.gridworld import get_var

MODEL_PATH = Path("models/ppo_200k_rho07_seed0.zip")
OUTPUT_PATH = Path("data/results/rho_sweep.json")

RHOS = (0.0, 0.3, 0.5, 0.7, 0.9, 1.0)
TRAIN_SEEDS = (0, 1, 2)

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
TARGET_LAMBDA = 0.1
INT_LAMBDA = 0.001
INT_REG = 0.01
BIAS = 1.0


def run_sparse(splits: dict, seed: int) -> dict:
    return run_sparse_gate(
        splits,
        epochs=EPOCHS,
        lr=LR,
        hidden=HIDDEN,
        batch_size=BATCH_SIZE,
        l1=L1,
        seed=seed,
    )


def run_casn_model(splits: dict, seed: int) -> dict:
    return run_casn(
        splits,
        epochs=EPOCHS,
        lr=LR,
        hidden=HIDDEN,
        rep_dim=REP_DIM,
        batch_size=BATCH_SIZE,
        l1=L1,
        target_lambda=TARGET_LAMBDA,
        int_lambda=INT_LAMBDA,
        int_reg=INT_REG,
        bias=BIAS,
        seed=seed,
    )


def fraction_s2_equals_s5(dataset: dict[str, np.ndarray]) -> float:
    s2 = np.asarray(
        [get_var(row, "s2").item() for row in dataset["obs"]]
    )
    s5 = np.asarray(
        [get_var(row, "s5").item() for row in dataset["obs"]]
    )

    return float(np.mean(s2 == s5))


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

        frac_equal = fraction_s2_equals_s5(dataset)

        seed_results = []

        for seed in TRAIN_SEEDS:
            print(f"  seed={seed}")

            sparse = run_sparse(splits, seed)
            casn = run_casn_model(splits, seed)

            seed_result = {
                "seed": seed,
                "sparse_gate": sparse,
                "casn": casn,
            }

            seed_results.append(seed_result)

            print(
                f"    sparse F1={sparse['recovery']['f1']:.3f} "
                f"mask={sparse['mask']}"
            )
            print(
                f"    CaSN    F1={casn['recovery']['f1']:.3f} "
                f"mask={casn['mask']}"
            )

        sparse_f1 = np.asarray(
            [
                result["sparse_gate"]["recovery"]["f1"]
                for result in seed_results
            ]
        )
        casn_f1 = np.asarray(
            [
                result["casn"]["recovery"]["f1"]
                for result in seed_results
            ]
        )

        result = {
            "rho": rho,
            "n_rows": len(dataset["Y"]),
            "n_episodes": N_EPISODES,
            "y_rate": float(dataset["Y"].mean()),
            "frac_s2_eq_s5": frac_equal,
            "seed_results": seed_results,
            "sparse_gate_mean_f1": float(sparse_f1.mean()),
            "sparse_gate_std_f1": float(sparse_f1.std()),
            "casn_mean_f1": float(casn_f1.mean()),
            "casn_std_f1": float(casn_f1.std()),
        }

        results.append(result)

        print(
            f"  frac(s2==s5)={frac_equal:.3f}"
        )
        print(
            f"  sparse mean F1="
            f"{sparse_f1.mean():.3f} ± {sparse_f1.std():.3f}"
        )
        print(
            f"  CaSN mean F1="
            f"{casn_f1.mean():.3f} ± {casn_f1.std():.3f}"
        )
        print()

    artifact = {
        "config": {
            "model": str(MODEL_PATH),
            "rhos": list(RHOS),
            "train_seeds": list(TRAIN_SEEDS),
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
            "target_lambda": TARGET_LAMBDA,
            "int_lambda": INT_LAMBDA,
            "int_reg": INT_REG,
            "bias": BIAS,
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
