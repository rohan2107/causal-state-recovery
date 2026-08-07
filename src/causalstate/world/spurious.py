from __future__ import annotations

import numpy as np


def corrupt(s2: int, rho: float, rng: np.random.Generator) -> int:
    if not (0 <= rho <= 1):
        raise ValueError(f"rho must be in [0, 1], got {rho}")

    if s2 not in (0, 1):
        raise ValueError(f"s2 must be 0 or 1, got {s2}")

    if rng.random() < rho:
        return s2
    else:
        return 1 - s2
