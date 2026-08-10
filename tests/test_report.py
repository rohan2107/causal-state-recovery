import numpy as np

from causalstate.evaluation.report import report
from causalstate.observational.mlp import MLP, make_dataloader


def test_report_returns_expected_results():
    rng = np.random.default_rng(0)

    split = {
        "obs": rng.random(
            (20, 9),
            dtype=np.float32,
        ),
        "Y": rng.integers(
            0,
            2,
            20,
        ),
    }

    model = MLP()

    _, scaler = make_dataloader(
        split,
    )

    scores = {
        "s0": 0.8,
        "s1": 0.7,
        "s2": 0.6,
        "s3": 0.5,
        "s4": 0.05,
        "s5": 0.04,
    }

    result = report(
        scores,
        model,
        scaler,
        {
            "train": split,
            "test": split,
            "ood_test": split,
        },
        threshold=0.1,
    )

    assert result["mask"] == [
        "s0",
        "s1",
        "s2",
        "s3",
    ]

    assert result["separation"] == 0.45

    assert result["recovery"]["f1"] == 1.0

    assert set(result["accuracies"]) == {
        "train",
        "test",
        "ood_test",
    }

    for accuracy in result["accuracies"].values():
        assert 0.0 <= accuracy <= 1.0
