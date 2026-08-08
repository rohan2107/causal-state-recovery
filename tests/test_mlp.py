from pathlib import Path

import numpy as np
import pytest
import torch

from causalstate.observational.dataset import load_split
from causalstate.observational.mlp import (
    MLP,
    evaluate,
    fit,
    make_dataloader,
    train_mlp,
)

TRAIN = Path("data/observational/train.npz")
OOD = Path("data/observational/ood_test.npz")

def test_mlp_output_shape():

    model = MLP()

    x = torch.randn(32, 9)

    logits = model(x)

    assert logits.shape == (32, 1)

def test_make_dataloader_reuses_scaler():

    train = {
        "obs": np.random.rand(20, 9).astype(np.float32),
        "Y": np.random.randint(0, 2, 20),
    }

    test = {
        "obs": np.random.rand(10, 9).astype(np.float32),
        "Y": np.random.randint(0, 2, 10),
    }

    _, scaler = make_dataloader(train)

    _, scaler2 = make_dataloader(
        test,
        scaler=scaler,
    )

    assert scaler2 is scaler

def test_make_dataloader_shapes():

    split = {
        "obs": np.random.rand(12, 9).astype(np.float32),
        "Y": np.random.randint(0, 2, 12),
    }

    loader, _ = make_dataloader(split)

    X, y = next(iter(loader))

    assert X.shape[1] == 9
    assert y.shape[1] == 1

def test_fit_runs():

    rng = np.random.default_rng(0)

    split = {
        "obs": rng.random((64, 9), dtype=np.float32),
        "Y": rng.integers(0, 2, 64),
    }

    loader, _ = make_dataloader(
        split,
        shuffle=True,
    )

    model = MLP()

    fit(
        model,
        loader,
        epochs=1,
    )

def test_evaluate_returns_probability():

    rng = np.random.default_rng(0)

    split = {
        "obs": rng.random((32, 9), dtype=np.float32),
        "Y": rng.integers(0, 2, 32),
    }

    loader, _ = make_dataloader(split)

    model = MLP()

    acc = evaluate(
        model,
        loader,
    )

    assert 0.0 <= acc <= 1.0

@pytest.mark.data
@pytest.mark.skipif(
    not TRAIN.exists(),
    reason="run scripts/build_dataset.py",
)
def test_train_mlp_is_reproducible():

    train = load_split(TRAIN)

    model1, scaler1 = train_mlp(
        train,
        seed=0,
    )

    loader1, _ = make_dataloader(
        train,
        scaler=scaler1,
    )

    acc1 = evaluate(
        model1,
        loader1,
    )

    model2, scaler2 = train_mlp(
        train,
        seed=0,
    )

    loader2, _ = make_dataloader(
        train,
        scaler=scaler2,
    )

    acc2 = evaluate(
        model2,
        loader2,
    )

    assert acc1 == pytest.approx(acc2)

@pytest.mark.data
@pytest.mark.skipif(
    not TRAIN.exists(),
    reason="run scripts/build_dataset.py",
)
def test_mlp_reaches_expected_accuracy():

    train = load_split(TRAIN)
    ood = load_split(OOD)

    model, scaler = train_mlp(
        train,
        seed=0,
    )

    train_loader, _ = make_dataloader(
        train,
        scaler=scaler,
    )

    ood_loader, _ = make_dataloader(
        ood,
        scaler=scaler,
    )

    train_acc = evaluate(
        model,
        train_loader,
    )

    ood_acc = evaluate(
        model,
        ood_loader,
    )

    assert train_acc > 0.90
    assert ood_acc > 0.85
