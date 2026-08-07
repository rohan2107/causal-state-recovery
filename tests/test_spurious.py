import numpy as np
import pytest

from causalstate.world.spurious import corrupt


def test_invalid_rho():
    rng = np.random.default_rng(0)

    with pytest.raises(ValueError):
        corrupt(0, -0.1, rng)

    with pytest.raises(ValueError):
        corrupt(0, 1.1, rng)

def test_invalid_s2():
    rng = np.random.default_rng(0)

    with pytest.raises(ValueError):
        corrupt(2, 0.7, rng)

    with pytest.raises(ValueError):
        corrupt(-1, 0.7, rng)

@pytest.mark.parametrize("rho", [0.0, 0.3, 0.5, 0.7, 0.9, 1.0])
@pytest.mark.parametrize("s2", [0, 1])
def test_match_rate(rho, s2):
    rng = np.random.default_rng(42)

    n = 20_000

    matches = sum(
        corrupt(s2, rho, rng) == s2
        for _ in range(n)
    )

    frac = matches / n

    if rho in (0.0, 1.0):
        assert frac == rho
    else:
        assert abs(frac - rho) < 0.02
