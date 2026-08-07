import hashlib
from pathlib import Path

import numpy as np

EPS_POOL = (0.15, 0.30, 0.45)

def content_hash(split: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()

    for key in sorted(split):
        h.update(key.encode())
        h.update(np.ascontiguousarray(split[key]).tobytes())

    return h.hexdigest()[:16]

def save_split(
    path: str | Path,
    split: dict[str, np.ndarray],
) -> None:
    np.savez_compressed(path, **split)

def load_split(
    path: str | Path,
) -> dict[str, np.ndarray]:

    data = np.load(path)

    return {k: data[k] for k in data.files}

def collect_observational(
    model,
    env,
    n_episodes: int,
    eps_pool: tuple[float, ...] = EPS_POOL,
    start_seed: int = 777,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:

    if rng is None:
        rng = np.random.default_rng(0)

    obs_rows = []
    y_rows = []
    eps_rows = []
    episode_rows = []
    step_rows = []

    for episode_id in range(n_episodes):

        eps = float(rng.choice(eps_pool))

        obs, _ = env.reset(
            seed=start_seed + episode_id,
        )

        trajectory = []
        done = False
        info = {}

        while not done:

            trajectory.append(obs.copy())

            if rng.random() < eps:
                action = int(rng.integers(env.action_space.n))
            else:
                action, _ = model.predict(
                    obs,
                    deterministic=True,
                )

            obs, _, terminated, truncated, info = env.step(
                int(action)
            )

            done = terminated or truncated

        y = int(info["Y"])

        for step, observation in enumerate(trajectory):
            obs_rows.append(observation)
            y_rows.append(y)
            eps_rows.append(eps)
            episode_rows.append(episode_id)
            step_rows.append(step)

    return {
        "obs": np.asarray(obs_rows),
        "Y": np.asarray(y_rows, dtype=np.int8),
        "eps": np.asarray(eps_rows, dtype=np.float32),
        "episode_id": np.asarray(
            episode_rows,
            dtype=np.int32,
        ),
        "step": np.asarray(
            step_rows,
            dtype=np.int32,
        ),
    }

def split_by_episode(
    dataset: dict[str, np.ndarray],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 0,
) -> dict[str, dict[str, np.ndarray]]:

    rng = np.random.default_rng(seed)

    episode_ids = np.unique(dataset["episode_id"])
    rng.shuffle(episode_ids)

    n = len(episode_ids)

    train_end = int(train_frac * n)
    val_end = train_end + int(val_frac * n)

    split_ids = {
        "train": episode_ids[:train_end],
        "val": episode_ids[train_end:val_end],
        "test": episode_ids[val_end:],
    }

    splits = {}

    for name, ids in split_ids.items():

        mask = np.isin(dataset["episode_id"], ids)

        splits[name] = {
            key: value[mask]
            for key, value in dataset.items()
        }

    return splits
