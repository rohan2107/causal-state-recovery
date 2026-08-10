import json
from pathlib import Path

from causalstate.observational.dataset import load_split
from causalstate.observational.sparse_gate import run_sparse_gate

DATA_DIR = Path("data/observational")
OUTPUT = DATA_DIR / "sparse_gate.json"


def main() -> None:
    splits = {
        name: load_split(DATA_DIR / filename)
        for name, filename in {
            "train": "train.npz",
            "test": "test.npz",
            "ood_test": "ood_test.npz",
        }.items()
    }

    result = run_sparse_gate(
        splits,
        epochs=40,
        lr=1e-3,
        hidden=64,
        batch_size=256,
        l1=0.02,
        seed=0,
    )

    OUTPUT.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
