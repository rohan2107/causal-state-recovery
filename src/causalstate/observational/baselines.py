import numpy as np
from sklearn.feature_selection import mutual_info_classif

from causalstate.evaluation.metrics import recovery_score, select
from causalstate.world.gridworld import CAUSAL_ORACLE, get_var

SUMMARY_VARS = ("s1", "s2", "s4", "s5")
CAUSAL_SUMMARY = CAUSAL_ORACLE & set(SUMMARY_VARS)

CORR_THRESHOLD = 0.1
MI_THRESHOLD = 0.01
MI_RANDOM_STATE = 0


def episode_summaries(
    split: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:

    episode_ids = np.unique(split["episode_id"])

    X = np.zeros(
        (len(episode_ids), len(SUMMARY_VARS)),
        dtype=np.float32,
    )

    y = np.zeros(
        len(episode_ids),
        dtype=np.int8,
    )

    for i, episode_id in enumerate(episode_ids):

        mask = split["episode_id"] == episode_id

        obs = split["obs"][mask]

        y[i] = split["Y"][mask][0]

        for j, var in enumerate(SUMMARY_VARS):

            values = np.array(
                [
                    get_var(row, var).item()
                    for row in obs
                ],
                dtype=np.float32,
            )

            X[i, j] = values.mean()

    return X, y


def correlation_scores(
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:

    scores = {}

    for i, var in enumerate(SUMMARY_VARS):

        if (
            np.std(X[:, i]) == 0
            or np.std(y) == 0
        ):
            scores[var] = 0.0
            continue

        corr = np.corrcoef(
            X[:, i],
            y,
        )[0, 1]

        scores[var] = abs(float(corr))

    return scores


def mi_scores(
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:

    mi = mutual_info_classif(
        X,
        y,
        random_state=MI_RANDOM_STATE,
    )

    return {
        var: float(score)
        for var, score in zip(SUMMARY_VARS, mi, strict=True)
    }

def run_baselines(
    split: dict[str, np.ndarray],
) -> dict:

    X, y = episode_summaries(split)

    corr = correlation_scores(X, y)
    mi = mi_scores(X, y)

    corr_selected = select(
        corr,
        CORR_THRESHOLD,
    )

    mi_selected = select(
        mi,
        MI_THRESHOLD,
    )

    return {
        "correlation": corr,
        "mutual_information": mi,
        "corr_selected": corr_selected,
        "mi_selected": mi_selected,
        "corr_recovery": recovery_score(
            corr_selected,
            oracle=CAUSAL_SUMMARY,
        ),
        "mi_recovery": recovery_score(
            mi_selected,
            oracle=CAUSAL_SUMMARY,
        ),
        "n_episodes": len(y),
    }
