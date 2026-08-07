from math import isnan

import pytest

from causalstate.oracle.recover import margin, recover, recovered_mask


def test_recovered_mask_layout2():
    scores = {
        "s0": 0.96,
        "s1": 1.0,
        "s2": 1.0,
        "s3": 0.64,
        "s4": 0.0,
        "s5": 0.0,
    }
    tau = 0.5
    expected = {"s0", "s1", "s2", "s3"}
    assert recovered_mask(scores, tau) == expected

def test_recovered_mask_higher_threshold():
    scores = {
        "s0": 0.96,
        "s1": 1.0,
        "s2": 1.0,
        "s3": 0.64,
        "s4": 0.0,
        "s5": 0.0,
    }

    expected = {"s0", "s1", "s2"}

    assert recovered_mask(scores, tau=0.7) == expected

def test_recovered_mask_layout13():
    scores = {
        "s0": 0.96,
        "s1": 1.0,
        "s2": 1.0,
        "s3": 0.51,
        "s4": 0.0,
        "s5": 0.0,
    }

    expected = {"s0", "s1", "s2", "s3"}

    assert recovered_mask(scores, tau=0.5) == expected

def test_recovered_mask_all_zero():
    scores = {
        "s0": 0.0,
        "s1": 0.0,
        "s2": 0.0,
        "s3": 0.0,
        "s4": 0.0,
        "s5": 0.0,
    }

    assert recovered_mask(scores) == set()

def test_recovered_mask_threshold_boundary():
    scores = {"s0": 0.5}

    assert recovered_mask(scores, tau=0.5) == set()

def test_recovered_mask_unknown_variable():
    with pytest.raises(ValueError):
        recovered_mask({"unknown_variable": 1.0})

def test_margin_layout2():
    scores = {
        "s0": 0.96,
        "s1": 1.0,
        "s2": 1.0,
        "s3": 0.64,
        "s4": 0.0,
        "s5": 0.0,
    }

    mask = {"s0","s1","s2","s3"}

    assert margin(scores, mask) == 0.64
    assert isnan(margin(scores, set()))
    assert isnan(margin(scores, set(scores)))

@pytest.mark.slow
def test_recover_pipeline(trained_model):
    result = recover(
        trained_model,
        base_seed=0,
        layout_seed=2,
        rho=0.7,
    )

    assert result["mask"] == {
        "s0",
        "s1",
        "s2",
        "s3",
    }

    recovery = result["recovery"]

    assert recovery["precision"] == 1.0
    assert recovery["recall"] == 1.0
    assert recovery["f1"] == 1.0

    assert recovery["false_positives"] == []
    assert recovery["false_negatives"] == []

    assert result["margin"] > 0.4

def test_recover_passes_layout_seed(monkeypatch):
    captured = {}

    def fake_make_env(**kwargs):
        captured.update(kwargs)

        class DummyEnv:
            def close(self):
                pass

        return DummyEnv()

    monkeypatch.setattr(
        "causalstate.oracle.recover.make_env",
        fake_make_env,
    )

    monkeypatch.setattr(
        "causalstate.oracle.recover.pn_sweep",
        lambda *args, **kwargs: {},
    )

    monkeypatch.setattr(
        "causalstate.oracle.recover.recovery_score",
        lambda mask: {},
    )

    recover(
        model=None,
        layout_seed=9,
    )

    assert captured["layout_seed"] == 9
