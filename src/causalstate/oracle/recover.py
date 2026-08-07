from math import nan

from causalstate.evaluation.metrics import recovery_score
from causalstate.oracle.replay_pn import RECOVERY_VARS, pn_sweep
from causalstate.world.envs import make_env

DEFAULT_TAU = 0.5

def recovered_mask(
    scores: dict[str, float],
    tau: float = DEFAULT_TAU,
) -> set[str]:
    mask = set()
    for var, score in scores.items():
        if var not in RECOVERY_VARS:
            raise ValueError(f"Unknown recovery variable: {var}")
        if score > tau:
            mask.add(var)
    return mask

def margin(
    scores: dict[str, float],
    mask: set[str],
) -> float:
    kept = [score for var, score in scores.items() if var in mask]
    rejected = [score for var, score in scores.items() if var not in mask]
    if not kept or not rejected:
        return nan
    return min(kept) - max(rejected)

def recover(
    model,
    n_episodes: int = 100,
    tau: float = DEFAULT_TAU,
    base_seed: int = 0,
    layout_seed: int = 2,
    rho: float | None = None,
):
    env = make_env(
        layout_seed=layout_seed,
        rho=rho,
    )

    try:
        scores = pn_sweep(
            model,
            env,
            n_episodes=n_episodes,
            base_seed=base_seed,
        )
    finally:
        env.close()

    mask = recovered_mask(scores, tau)

    score = recovery_score(mask)

    gap = margin(scores, mask)

    return {
        "scores": scores,
        "mask": mask,
        "recovery": score,
        "margin": gap,
    }