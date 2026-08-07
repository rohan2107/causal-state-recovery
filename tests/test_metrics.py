import pytest

from causalstate.evaluation.metrics import recovery_score
from causalstate.world.gridworld import CAUSAL_ORACLE


def test_recovery_score_perfect():
    score = recovery_score(CAUSAL_ORACLE)

    assert score["precision"] == 1.0
    assert score["recall"] == 1.0
    assert score["f1"] == 1.0
    assert score["true_positives"] == sorted(CAUSAL_ORACLE)
    assert score["false_positives"] == []
    assert score["false_negatives"] == []

def test_recovery_score_with_false_positive():
    score = recovery_score(
        {"s0", "s1", "s2", "s3", "s5"}
    )

    assert score["precision"] == pytest.approx(0.8)
    assert score["recall"] == 1.0
    assert score["f1"] == pytest.approx(8 / 9)

    assert score["false_positives"] == ["s5"]
    assert score["false_negatives"] == []

def test_recovery_score_partial():
    score = recovery_score(
        {"s1", "s2", "s5"}
    )

    assert score["precision"] == pytest.approx(2 / 3)
    assert score["recall"] == 0.5
    assert score["f1"] == pytest.approx(4 / 7)

    assert score["true_positives"] == ["s1", "s2"]
    assert score["false_positives"] == ["s5"]
    assert score["false_negatives"] == ["s0", "s3"]

def test_recovery_score_all_wrong():
    score = recovery_score(
        {"s4", "s5"}
    )

    assert score["precision"] == 0.0
    assert score["recall"] == 0.0
    assert score["f1"] == 0.0

    assert score["true_positives"] == []
    assert score["false_positives"] == ["s4", "s5"]
    assert score["false_negatives"] == sorted(CAUSAL_ORACLE)

def test_recovery_score_empty():
    score = recovery_score(set())

    assert score["precision"] == 0.0
    assert score["recall"] == 0.0
    assert score["f1"] == 0.0

    assert score["true_positives"] == []
    assert score["false_positives"] == []
    assert score["false_negatives"] == sorted(CAUSAL_ORACLE)

def test_recovery_score_all_variables():
    score = recovery_score(
        {"s0", "s1", "s2", "s3", "s4", "s5"}
    )

    assert score["precision"] == pytest.approx(2 / 3)
    assert score["recall"] == 1.0
    assert score["f1"] == pytest.approx(4 / 5)

    assert score["false_positives"] == ["s4", "s5"]

def test_recovery_score_unknown_variable():
    with pytest.raises(ValueError):
        recovery_score({"s0", "s10"})

def test_recovery_score_accepts_generator():
    score = recovery_score(
        v for v in CAUSAL_ORACLE
    )

    assert score["f1"] == 1.0

def test_recovery_score_custom_oracle():
    score = recovery_score(
        {"s0"},
        oracle={"s0"},
    )

    assert score["precision"] == 1.0
    assert score["recall"] == 1.0
    assert score["f1"] == 1.0
