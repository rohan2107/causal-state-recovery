import json
from pathlib import Path

import numpy as np

from causalstate.observational.baselines import (
    SUMMARY_VARS,
    episode_summaries,
    mi_scores,
)
from causalstate.observational.dataset import load_split

DATA_PATH = Path("data/observational/baseline_summaries_seed0.npz")
OUTPUT_PATH = Path("data/results/ablation_fixed_length.json")

K_VALUES = (9, 15, 20, 30, 50)
NULL_SHUFFLES = 30
NULL_SEED = 0


def permutation_null(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_shuffles: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)

    null_scores = {
        var: []
        for var in SUMMARY_VARS
    }

    for _ in range(n_shuffles):
        shuffled_y = rng.permutation(y)
        scores = mi_scores(X, shuffled_y)

        for var in SUMMARY_VARS:
            null_scores[var].append(scores[var])

    return {
        var: float(np.percentile(values, 95))
        for var, values in null_scores.items()
    }


def main() -> None:
    split = load_split(DATA_PATH)

    X_full, y_full = episode_summaries(split)
    full_mi = mi_scores(X_full, y_full)

    results = []

    for k in K_VALUES:
        X_fixed, y_fixed = episode_summaries(
            split,
            max_steps=k,
        )

        if not np.array_equal(y_full, y_fixed):
            raise RuntimeError(
                f"Labels changed for K={k}."
            )

        fixed_mi = mi_scores(
            X_fixed,
            y_fixed,
        )

        null_p95 = permutation_null(
            X_fixed,
            y_fixed,
            n_shuffles=NULL_SHUFFLES,
            seed=NULL_SEED,
        )

        result = {
            "k": k,
            "mutual_information": fixed_mi,
            "null_p95": null_p95,
        }

        results.append(result)

    artifact = {
        "config": {
            "dataset": str(DATA_PATH),
            "n_episodes": len(y_full),
            "k_values": list(K_VALUES),
            "null_shuffles": NULL_SHUFFLES,
            "null_seed": NULL_SEED,
        },
        "full": {
            "mutual_information": full_mi,
        },
        "fixed_length": results,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            artifact,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=== Fixed-length MI K-sweep ===")
    print(f"episodes: {len(y_full)}")
    print()

    print(
        f"{'K':>5}"
        f"{'s1':>10}"
        f"{'s2':>10}"
        f"{'s4':>10}"
        f"{'s5':>10}"
        f"{'s4 null95':>12}"
    )

    for result in results:
        mi = result["mutual_information"]
        null = result["null_p95"]

        print(
            f"{result['k']:>5}"
            f"{mi['s1']:>10.3f}"
            f"{mi['s2']:>10.3f}"
            f"{mi['s4']:>10.3f}"
            f"{mi['s5']:>10.3f}"
            f"{null['s4']:>12.3f}"
        )

    print()
    print("Full-length:")
    for var in SUMMARY_VARS:
        print(f"  {var}: {full_mi[var]:.3f}")

    print()
    print(f"Wrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
