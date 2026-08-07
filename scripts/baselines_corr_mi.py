import json
from pathlib import Path

from causalstate.observational.baselines import (
    CORR_THRESHOLD,
    MI_THRESHOLD,
    run_baselines,
)
from causalstate.observational.dataset import load_split

DATASET = Path("data/observational/train.npz")
OUTPUT = Path(
    "data/observational/baselines.json"
)


def main():

    split = load_split(DATASET)

    result = run_baselines(split)
    artifact = result.copy()

    artifact["corr_selected"] = sorted(
        artifact["corr_selected"]
    )

    artifact["mi_selected"] = sorted(
        artifact["mi_selected"]
    )

    with open(
        OUTPUT,
        "w",
    ) as f:
        json.dump(
            artifact,
            f,
            indent=2,
        )

    print("\nCorrelation\n")

    for var, score in result["correlation"].items():
        print(f"{var:>2}: {score:.3f}")

    print(
        f"\nSelected (τ={CORR_THRESHOLD:.2f}): "
        f"{result['corr_selected']}"
    )

    print(
        f"Recovery F1 : "
        f"{result['corr_recovery']['f1']:.3f}"
    )

    print("\nMutual Information\n")

    for var, score in result["mutual_information"].items():
        print(f"{var:>2}: {score:.3f}")

    print(
        f"\nSelected (τ={MI_THRESHOLD:.2f}): "
        f"{result['mi_selected']}"
    )

    print(
        f"Recovery F1 : "
        f"{result['mi_recovery']['f1']:.3f}"
    )

    print(
        f"\nEpisodes: {result['n_episodes']}"
    )


if __name__ == "__main__":
    main()
