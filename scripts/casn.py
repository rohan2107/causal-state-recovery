import json
from pathlib import Path

from causalstate.observational.casn import run_casn
from causalstate.observational.dataset import load_split

DATA_DIR = Path("data/observational")
OUTPUT = DATA_DIR / "casn.json"


def main() -> None:
    splits = {
        name: load_split(DATA_DIR / filename)
        for name, filename in {
            "train": "train.npz",
            "test": "test.npz",
            "ood_test": "ood_test.npz",
        }.items()
    }

    result = run_casn(
        splits,
        epochs=40,
        lr=1e-3,
        hidden=64,
        rep_dim=32,
        batch_size=256,
        l1=0.02,
        target_lambda=0.1,
        int_lambda=0.001,
        int_reg=0.01,
        bias=1.0,
        seed=0,
    )

    OUTPUT.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
