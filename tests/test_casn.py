from pathlib import Path

import numpy as np
import pytest
import torch

from causalstate.evaluation.metrics import recovery_score, select
from causalstate.observational.casn import (
    CaSN,
    intervention_magnitude,
    intervention_regularization,
    necessity_loss,
    run_casn,
    train_casn,
)
from causalstate.observational.dataset import load_split
from causalstate.observational.mlp import make_dataloader
from causalstate.observational.sparse_gate import GATE_TAU, gate_values, train_gate
from causalstate.world.gridworld import ALL_VARS, CAUSAL_ORACLE, VAR_SLICES

TRAIN = Path("data/observational/train.npz")

def test_casn_uses_shared_classifier():
    model = CaSN()

    assert model.classifier is not None

    x = torch.randn(8, 9)

    y, int_y, delta, z = model.forward_all(x)

    assert y.shape == (8, 1)
    assert int_y.shape == (8, 1)
    assert delta.shape == (8, 32)
    assert z.shape == (8, 32)


def test_casn_intervention_zero_matches_original_prediction():
    model = CaSN()

    x = torch.randn(8, 9)

    with torch.no_grad():
        z = model.encode(x)
        y = model.classifier(z)

        int_y = model.classifier(z)

    assert torch.allclose(y, int_y)


def test_casn_gate_values_returns_all_variables():
    model = CaSN()

    scores = model.gate_values()

    assert set(scores) == set(ALL_VARS)

    expected = torch.sigmoid(
        torch.tensor(2.0)
    ).item()

    for var in ALL_VARS:
        assert scores[var] == pytest.approx(expected)


def test_casn_gate_expansion_matches_var_slices():
    model = CaSN()

    with torch.no_grad():
        model.gate_logits.fill_(10.0)

        s0_idx = ALL_VARS.index("s0")
        model.gate_logits[s0_idx] = -10.0

    gate = model._expanded_gate()

    for var in ALL_VARS:
        sl = VAR_SLICES[var]

        if var == "s0":
            assert torch.all(gate[sl] < 1e-3)
        else:
            assert torch.all(gate[sl] > 0.999)

def test_necessity_loss_is_more_negative_when_predictions_disagree():
    y_logits = torch.tensor([[10.0]])
    matching_int_logits = torch.tensor([[10.0]])
    different_int_logits = torch.tensor([[-10.0]])

    matching = necessity_loss(
        y_logits,
        matching_int_logits,
    )

    different = necessity_loss(
        y_logits,
        different_int_logits,
    )

    assert different < matching

def test_intervention_regularization_prefers_unit_magnitude():
    zero = torch.zeros(1, 32)
    unit = torch.ones(1, 32)

    zero_penalty = intervention_regularization(zero)
    unit_penalty = intervention_regularization(unit)

    assert zero_penalty > unit_penalty

@pytest.mark.data
@pytest.mark.skipif(
    not TRAIN.exists(),
    reason="run scripts/build_dataset.py",
)
def test_intervention_is_functional():
    train = load_split(TRAIN)

    model, scaler = train_casn(
        train,
        seed=0,
    )

    loader, _ = make_dataloader(
        train,
        scaler=scaler,
    )

    magnitude = intervention_magnitude(
        model,
        loader,
    )

    assert magnitude > 5.0

def test_run_casn_returns_report():
    rng = np.random.default_rng(0)

    split = {
        "obs": rng.random((32, 9), dtype=np.float32),
        "Y": rng.integers(0, 2, 32),
    }

    result = run_casn(
        {
            "train": split,
            "test": split,
            "ood_test": split,
        },
        epochs=1,
        seed=0,
    )

    assert set(result) == {
        "scores",
        "separation",
        "mask",
        "recovery",
        "accuracies",
    }

@pytest.mark.data
@pytest.mark.skipif(
    not TRAIN.exists(),
    reason="run scripts/build_dataset.py",
)
def test_target_lambda_does_not_raise_s5_monotonically():
    train = load_split(TRAIN)

    s5_values = []

    for target_lambda in (0.1, 0.5, 2.0):
        model, _ = train_casn(
            train,
            target_lambda=target_lambda,
            seed=0,
        )

        s5_values.append(
            model.gate_values()["s5"]
        )

    assert s5_values[-1] <= s5_values[0]

@pytest.mark.data
@pytest.mark.skipif(
    not TRAIN.exists(),
    reason="run scripts/build_dataset.py",
)
def test_train_casn_is_reproducible():
    train = load_split(TRAIN)

    model1, _ = train_casn(
        train,
        seed=0,
    )

    model2, _ = train_casn(
        train,
        seed=0,
    )

    assert torch.allclose(
        model1.gate_logits,
        model2.gate_logits,
    )

@pytest.mark.data
@pytest.mark.skipif(
    not TRAIN.exists(),
    reason="run scripts/build_dataset.py",
)
def test_casn_matches_l1_gate():
    train = load_split(TRAIN)

    casn_model, _ = train_casn(
        train,
        seed=0,
    )
    casn_scores = casn_model.gate_values()
    casn_mask = select(casn_scores, GATE_TAU)
    casn_recovery = recovery_score(
        casn_mask,
        oracle=CAUSAL_ORACLE,
    )

    l1_model, _ = train_gate(
        train,
        seed=0,
    )
    l1_scores = gate_values(l1_model)
    l1_mask = select(l1_scores, GATE_TAU)
    l1_recovery = recovery_score(
        l1_mask,
        oracle=CAUSAL_ORACLE,
    )

    assert casn_mask == set(CAUSAL_ORACLE)
    assert l1_mask == set(CAUSAL_ORACLE)

    assert casn_recovery["f1"] == pytest.approx(1.0)
    assert l1_recovery["f1"] == pytest.approx(1.0)
