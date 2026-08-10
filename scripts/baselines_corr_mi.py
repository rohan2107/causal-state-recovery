import json
from pathlib import Path

from causalstate.observational.baselines import (
    CORR_THRESHOLD,
    MI_THRESHOLD,
    run_baselines,
)
from causalstate.observational.dataset import load_split

DATASETS = tuple(
    Path(
        f"data/observational/baseline_summaries_seed{seed}.npz"
    )
    for seed in (0, 1, 2)
)

OUTPUT = Path(
    "data/observational/baselines.json"
)


def main() -> None:
    results = [
        run_baselines(load_split(dataset))
        for dataset in DATASETS
    ]

    artifact = {
        "seeds": [0, 1, 2],
        "n_episodes": [
            result["n_episodes"]
            for result in results
        ],
        "correlation": {
            var: [
                result["correlation"][var]
                for result in results
            ]
            for var in results[0]["correlation"]
        },
        "mutual_information": {
            var: [
                result["mutual_information"][var]
                for result in results
            ]
            for var in results[0]["mutual_information"]
        },
        "corr_selected": [
            sorted(result["corr_selected"])
            for result in results
        ],
        "mi_selected": [
            sorted(result["mi_selected"])
            for result in results
        ],
        "corr_recovery": [
            result["corr_recovery"]
            for result in results
        ],
        "mi_recovery": [
            result["mi_recovery"]
            for result in results
        ],
    }

    OUTPUT.write_text(
        json.dumps(
            artifact,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nCorrelation\n")

    for var, scores in artifact["correlation"].items():
        print(
            f"{var:>2}: "
            f"{min(scores):.3f}–{max(scores):.3f}"
        )

    print(
        f"\nSelected (τ={CORR_THRESHOLD:.2f}): "
        f"{artifact['corr_selected']}"
    )

    print("\nMutual Information\n")

    for var, scores in artifact["mutual_information"].items():
        print(
            f"{var:>2}: "
            f"{min(scores):.3f}–{max(scores):.3f}"
        )

    print(
        f"\nSelected (τ={MI_THRESHOLD:.2f}): "
        f"{artifact['mi_selected']}"
    )

    print(
        f"\nEpisodes per seed: "
        f"{artifact['n_episodes']}"
    )


if __name__ == "__main__":
    main()
