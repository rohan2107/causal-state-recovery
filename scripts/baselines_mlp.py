import json
from pathlib import Path

from causalstate.observational.dataset import load_split
from causalstate.observational.mlp import (
    evaluate,
    make_dataloader,
    train_mlp,
)

TRAIN = Path("data/observational/train.npz")
VAL = Path("data/observational/val.npz")
TEST = Path("data/observational/test.npz")
OOD = Path("data/observational/ood_test.npz")
OUTPUT = OUTPUT = Path("data/observational/mlp.json")

def main():
    train = load_split(TRAIN)
    val = load_split(VAL)
    test = load_split(TEST)
    ood = load_split(OOD)

    model, scaler = train_mlp(
        train,
        seed=0,
    )

    train_eval_loader, _ = make_dataloader(
        train,
        scaler=scaler,
    )

    val_loader, _ = make_dataloader(
        val,
        scaler=scaler,
    )

    test_loader, _ = make_dataloader(
        test,
        scaler=scaler,
    )

    ood_loader, _ = make_dataloader(
        ood,
        scaler=scaler,
    )

    train_acc = evaluate(model, train_eval_loader)
    val_acc = evaluate(model, val_loader)
    test_acc = evaluate(model, test_loader)
    ood_acc = evaluate(model, ood_loader)

    results = {
        "seed": 0,
        "epochs": 30,
        "hidden": 64,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "ood_accuracy": ood_acc,
    }

    with open(
        OUTPUT,
        "w",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print(f"Train : {train_acc:.3f}")
    print(f"Val   : {val_acc:.3f}")
    print(f"Test  : {test_acc:.3f}")
    print(f"OOD   : {ood_acc:.3f}")

if __name__ == "__main__":
    main()
