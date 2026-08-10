from pathlib import Path

import numpy as np
import pytest
import torch

from causalstate.evaluation.metrics import recovery_score, select, separation
from causalstate.observational.dataset import load_split
from causalstate.observational.sparse_gate import (
    DEFAULT_L1,
    GATE_TAU,
    GatedMLP,
    gate_penalty,
    gate_values,
    run_sparse_gate,
    train_gate,
)
from causalstate.world.gridworld import ALL_VARS, CAUSAL_ORACLE, VAR_SLICES

TRAIN = Path("data/observational/train.npz")


def test_expanded_gate_matches_var_slices():
    model = GatedMLP()

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

def test_gate_penalty_counts_variables_not_columns():
    model = GatedMLP()

    with torch.no_grad():
        model.gate_logits.fill_(2.0)

    expected = (
        DEFAULT_L1
        * len(ALL_VARS)
        * torch.sigmoid(torch.tensor(2.0))
    )

    penalty = DEFAULT_L1 * gate_penalty(model)

    assert torch.isclose(
        penalty,
        expected,
    )

def test_gate_initialisation():
    model = GatedMLP()

    scores = torch.sigmoid(model.gate_logits)

    expected = torch.full_like(
        scores,
        torch.sigmoid(torch.tensor(2.0)),
    )

    assert len(scores) == len(ALL_VARS)
    assert torch.allclose(scores, expected)

@pytest.mark.data
@pytest.mark.skipif(
    not TRAIN.exists(),
    reason="run scripts/build_dataset.py",
)
def test_train_gate_is_reproducible():
    train = load_split(TRAIN)

    model1, _ = train_gate(
        train,
        seed=0,
    )

    model2, _ = train_gate(
        train,
        seed=0,
    )

    assert torch.allclose(
        model1.gate_logits,
        model2.gate_logits,
    )

def test_separation_is_positive_when_causal_scores_are_higher():
    scores = {
        "s0": 0.8,
        "s1": 0.7,
        "s2": 0.6,
        "s3": 0.5,
        "s4": 0.2,
        "s5": 0.1,
    }

    assert separation(scores) > 0

def test_separation_is_negative_when_nuisance_scores_are_higher():
    scores = {
        "s0": 0.2,
        "s1": 0.3,
        "s2": 0.4,
        "s3": 0.1,
        "s4": 0.8,
        "s5": 0.7,
    }

    assert separation(scores) < 0

def test_gate_values_returns_all_variables():
    model = GatedMLP()

    scores = gate_values(model)

    assert set(scores) == set(ALL_VARS)

    expected = torch.sigmoid(
        torch.tensor(2.0)
    ).item()

    for var in ALL_VARS:
        assert scores[var] == pytest.approx(expected)

def test_run_sparse_gate_returns_report():
    rng = np.random.default_rng(0)

    split = {
        "obs": rng.random(
            (32, 9),
            dtype=np.float32,
        ),
        "Y": rng.integers(
            0,
            2,
            32,
        ),
    }

    result = run_sparse_gate(
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

    assert set(result["scores"]) == {
        "s0",
        "s1",
        "s2",
        "s3",
        "s4",
        "s5",
    }

    assert set(result["accuracies"]) == {
        "train",
        "test",
        "ood_test",
    }

@pytest.mark.data
@pytest.mark.skipif(
    not TRAIN.exists(),
    reason="run scripts/build_dataset.py",
)
def test_sparse_gate_recovers_causal_mask():
    train = load_split(TRAIN)

    model, _ = train_gate(
        train,
        seed=0,
    )

    scores = gate_values(model)
    mask = select(
        scores,
        GATE_TAU,
    )

    recovery = recovery_score(
        mask,
        oracle=CAUSAL_ORACLE,
    )

    assert mask == {
        "s0",
        "s1",
        "s2",
        "s3",
    }

    assert recovery["f1"] == pytest.approx(1.0)

    for var in CAUSAL_ORACLE:
        assert scores[var] > GATE_TAU

    for var in ALL_VARS:
        if var not in CAUSAL_ORACLE:
            assert scores[var] < GATE_TAU

@pytest.mark.data
@pytest.mark.slow
@pytest.mark.skipif(
    not TRAIN.exists(),
    reason="run scripts/build_dataset.py",
)
def test_sparse_gate_separation_is_positive_across_seeds_and_l1():
    train = load_split(TRAIN)

    for seed in (0, 1, 2):
        for l1 in (0.01, 0.02, 0.05):
            model, _ = train_gate(
                train,
                seed=seed,
                l1=l1,
            )

            scores = gate_values(model)
            gap = separation(scores)

            assert gap > 0, (
                f"separation failed for seed={seed}, "
                f"l1={l1}: {gap:.4f}"
            )

@pytest.mark.data
@pytest.mark.skipif(
    not TRAIN.exists(),
    reason="run scripts/build_dataset.py",
)
def test_remove_s2_increases_spurious_gate():
    train = load_split(TRAIN)

    baseline_model, _ = train_gate(
        train,
        seed=0,
    )
    baseline_scores = gate_values(baseline_model)

    ablated = {
        "obs": train["obs"].copy(),
        "Y": train["Y"].copy(),
    }

    ablated["obs"][:, 4] = 0.0

    ablated_model, _ = train_gate(
        ablated,
        seed=0,
    )
    ablated_scores = gate_values(ablated_model)

    assert ablated_scores["s5"] > baseline_scores["s5"]
    assert ablated_scores["s5"] - baseline_scores["s5"] > 0.03
