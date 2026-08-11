import json
from pathlib import Path

from stable_baselines3 import PPO

from causalstate.agent.ppo import success_rate
from causalstate.oracle.recover import recover
from causalstate.world.envs import make_env

LAYOUTS = (2, 4, 5, 9, 13)
RHO = 0.7
N_EPISODES = 200
COMPETENCE_THRESHOLD = 0.5
OUTPUT_PATH = Path("data/results/layout_robustness.json")

MODELS = {
    2: Path("models/ppo_200k_rho07_seed0.zip"),
    4: Path("models/ppo_200k_rho07_seed0_layout4.zip"),
    5: Path("models/ppo_200k_rho07_seed1_layout5.zip"),
    9: Path("models/ppo_200k_rho07_seed0_layout9.zip"),
    13: Path("models/ppo_200k_rho07_seed0_layout13.zip"),
}

TRAIN_SEEDS = {
    2: 0,
    4: 0,
    5: 1,
    9: 0,
    13: 0,
}


def geometry(layout_seed: int) -> dict:
    env = make_env(layout_seed=layout_seed, rho=RHO)
    env.reset(seed=0)

    fgw = env.unwrapped

    key = next(
        (x, y)
        for x in range(fgw.width)
        for y in range(fgw.height)
        if type(fgw.grid.get(x, y)).__name__ == "Key"
    )

    result = {
        "door": list(fgw._door_pos),
        "split_col": fgw._door_pos[0],
        "key": list(key),
        "s3_domain_size": len(env.intervention_domain("s3")),
        "s0_domain_size": len(env.intervention_domain("s0")),
    }

    env.close()
    return result


def main() -> None:
    results = []

    print("=== Layout robustness ===")

    for layout_seed in LAYOUTS:
        model_path = MODELS[layout_seed]
        model = PPO.load(model_path)

        success = success_rate(
            model,
            n_episodes=N_EPISODES,
            layout_seed=layout_seed,
            rho=RHO,
        )

        result = recover(
            model,
            n_episodes=N_EPISODES,
            layout_seed=layout_seed,
            rho=RHO,
        )

        competent = success > COMPETENCE_THRESHOLD

        row = {
            "layout_seed": layout_seed,
            "train_seed": TRAIN_SEEDS[layout_seed],
            "model": str(model_path),
            "geometry": geometry(layout_seed),
            "success_rate": success,
            "competent": competent,
            "scores": result["scores"],
            "mask": sorted(result["mask"]),
            "recovery": result["recovery"],
            "margin": result["margin"],
            "n_episodes": N_EPISODES,
            "rho": RHO,
            "competence_threshold": COMPETENCE_THRESHOLD,
        }

        results.append(row)

        print(
            f"layout={layout_seed} "
            f"seed={TRAIN_SEEDS[layout_seed]} "
            f"success={success:.3f} "
            f"s3={result['scores']['s3']:.3f} "
            f"mask={sorted(result['mask'])} "
            f"F1={result['recovery']['f1']:.3f} "
            f"margin={result['margin']:.3f}"
        )

    artifact = {
        "config": {
            "layouts": list(LAYOUTS),
            "rho": RHO,
            "n_episodes": N_EPISODES,
            "competence_threshold": COMPETENCE_THRESHOLD,
        },
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w") as f:
        json.dump(artifact, f, indent=2)

    print(f"\nWrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
