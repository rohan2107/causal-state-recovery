from causalstate.evaluation.metrics import recovery_score, select, separation
from causalstate.observational.mlp import evaluate, make_dataloader
from causalstate.world.gridworld import CAUSAL_ORACLE


def report(
    scores: dict[str, float],
    model,
    scaler,
    splits: dict[str, dict],
    *,
    threshold: float,
) -> dict:
    mask = select(
        scores,
        threshold,
    )

    recovery = recovery_score(
        mask,
        oracle=CAUSAL_ORACLE,
    )

    accuracies = {}

    for name, split in splits.items():
        loader, _ = make_dataloader(
            split,
            scaler=scaler,
        )

        accuracies[name] = evaluate(
            model,
            loader,
        )

    return {
        "scores": scores,
        "separation": separation(scores),
        "mask": sorted(mask),
        "recovery": recovery,
        "accuracies": accuracies,
    }
