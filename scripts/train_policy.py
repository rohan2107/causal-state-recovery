from pathlib import Path

from causalstate.agent.ppo import success_rate, train

MODEL_PATH = Path("models/ppo_200k_rho07_seed0.zip")

TOTAL_TIMESTEPS = 200_000
SEED = 0
LAYOUT_SEED = 2
RHO = 0.7
ENT_COEF = 0.05
EVAL_EVERY = 25_000
EVAL_EPISODES = 50


def main() -> None:
    model = train(
        total_timesteps=TOTAL_TIMESTEPS,
        seed=SEED,
        layout_seed=LAYOUT_SEED,
        rho=RHO,
        prune=False,
        save_path=MODEL_PATH,
        ent_coef=ENT_COEF,
        eval_every=EVAL_EVERY,
        eval_episodes=EVAL_EPISODES,
    )

    final_success = success_rate(
        model,
        n_episodes=100,
        layout_seed=LAYOUT_SEED,
        start_seed=123,
        rho=RHO,
        prune=False,
    )

    print("=== PPO training ===")
    print(f"timesteps: {TOTAL_TIMESTEPS}")
    print(f"seed: {SEED}")
    print(f"layout_seed: {LAYOUT_SEED}")
    print(f"rho: {RHO}")
    print(f"ent_coef: {ENT_COEF}")
    print(f"success_rate: {final_success:.3f}")
    print(f"model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
