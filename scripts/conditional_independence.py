from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from causalstate.evaluation.metrics import recovery_score
from causalstate.observational.baselines import (
    CAUSAL_SUMMARY,
    SUMMARY_VARS,
    episode_summaries,
)
from causalstate.world.gridworld import get_var

SEEDS = range(3)
RHO = 0.7
THRESHOLD = 0.1
TOL = 1e-12
OUTPUT_PATH = Path("data/results/conditional_independence.json")

CI_VARS = ("s1", "s2", "s4", "s5")


def partial_correlation(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    if z.ndim == 1:
        z = z[:, None]

    design = np.column_stack((np.ones(len(z)), z))

    x_res = x - design @ np.linalg.lstsq(
        design, x, rcond=None
    )[0]
    y_res = y - design @ np.linalg.lstsq(
        design, y, rcond=None
    )[0]

    if (
        not np.all(np.isfinite(x_res))
        or not np.all(np.isfinite(y_res))
        or np.std(x_res) <= TOL
        or np.std(y_res) <= TOL
    ):
        return float("nan")

    return float(np.corrcoef(x_res, y_res)[0, 1])


def scores(
    X: np.ndarray,
    y: np.ndarray,
    mode: str,
) -> dict[str, float]:
    if mode == "s2":
        vars_to_test = ("s1", "s4", "s5")
    elif mode == "all":
        vars_to_test = CI_VARS
    else:
        raise ValueError(mode)

    result = {}

    for var in vars_to_test:
        i = SUMMARY_VARS.index(var)

        if mode == "s2":
            z = X[:, SUMMARY_VARS.index("s2")]
        else:
            z = X[
                :,
                [
                    j
                    for j in range(len(SUMMARY_VARS))
                    if j != i
                ],
            ]

        result[var] = partial_correlation(
            X[:, i],
            y,
            z,
        )

    return result


def select(result: dict[str, float]) -> set[str]:
    return {
        var
        for var, value in result.items()
        if np.isfinite(value) and abs(value) > THRESHOLD
    }


def summary_stats(
    results: list[dict[str, float]],
    vars_to_test: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    output = {}

    for var in vars_to_test:
        values = np.array(
            [r[var] for r in results],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(values)):
            output[var] = {"mean": float("nan"), "std": float("nan")}
        else:
            output[var] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
            }

    return output


def main() -> None:
    episode_s2 = []
    episode_all = []
    step_s2 = []
    step_all = []

    print("=== Conditional independence ===")

    for seed in SEEDS:
        path = Path(
            f"data/observational/baseline_summaries_seed{seed}.npz"
        )
        split = dict(np.load(path))

        if "eps" in split:
            raise ValueError(
                f"{path} is epsilon-pooled; expected epsilon=0.45."
            )

        X_episode, y_episode = episode_summaries(split)

        obs = split["obs"]
        X_step = np.column_stack(
            [
                [
                    get_var(row, var).item()
                    for row in obs
                ]
                for var in SUMMARY_VARS
            ]
        )
        y_step = split["Y"]

        e_s2 = scores(X_episode, y_episode, "s2")
        e_all = scores(X_episode, y_episode, "all")
        s_s2 = scores(X_step, y_step, "s2")
        s_all = scores(X_step, y_step, "all")

        episode_s2.append(e_s2)
        episode_all.append(e_all)
        step_s2.append(s_s2)
        step_all.append(s_all)

        print(
            f"seed={seed} "
            f"episode[s2]={e_s2} "
            f"episode[all]={e_all}"
        )
        print(
            f"       step[s2]={s_s2} "
            f"step[all]={s_all}"
        )
        print(
            f"       masks: "
            f"episode[s2]={sorted(select(e_s2))} "
            f"episode[all]={sorted(select(e_all))}"
        )

    episode_s2_masks = [
        sorted(select(r)) for r in episode_s2
    ]
    episode_all_masks = [
        sorted(select(r)) for r in episode_all
    ]
    step_s2_masks = [
        sorted(select(r)) for r in step_s2
    ]
    step_all_masks = [
        sorted(select(r)) for r in step_all
    ]

    tested_recovery_s2 = [
        recovery_score(set(mask), oracle=frozenset({"s1"}))
        for mask in episode_s2_masks
    ]
    fully_conditioned_oracle = CAUSAL_SUMMARY
    tested_recovery_all = [
        recovery_score(set(mask), oracle=fully_conditioned_oracle)
        for mask in episode_all_masks
    ]

    artifact = {
        "config": {
            "seeds": list(SEEDS),
            "rho": RHO,
            "threshold": THRESHOLD,
            "conditioning": {
                "s2": {
                    "condition_on": ["s2"],
                    "test_vars": ["s1", "s4", "s5"],
                },
                "all": {
                    "condition_on": ["all_other_summary_variables"],
                    "test_vars": list(CI_VARS),
                },
            },
        },
        "episode": {
            "s2": {
                "per_seed": episode_s2,
                "mean_std": summary_stats(
                    episode_s2,
                    ("s1", "s4", "s5"),
                ),
                "masks": episode_s2_masks,
            },
            "all": {
                "per_seed": episode_all,
                "mean_std": summary_stats(
                    episode_all,
                    CI_VARS,
                ),
                "masks": episode_all_masks,
            },
        },
        "step": {
            "s2": {
                "per_seed": step_s2,
                "mean_std": summary_stats(
                    step_s2,
                    ("s1", "s4", "s5"),
                ),
                "masks": step_s2_masks,
            },
            "all": {
                "per_seed": step_all,
                "mean_std": summary_stats(
                    step_all,
                    CI_VARS,
                ),
                "masks": step_all_masks,
            },
        },
        "recovery": {
            "s2_conditioned_oracle": ["s1"],
            "fully_conditioned_oracle": sorted(
                fully_conditioned_oracle
            ),
            "s2_conditioned": tested_recovery_s2,
            "fully_conditioned": tested_recovery_all,
        },
        "rho_1": {
            "status": "undefined",
            "reason": "s5 equals s2 identically, so its residual given s2 is zero.",
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w") as f:
        json.dump(artifact, f, indent=2)

    print(f"\nWrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
