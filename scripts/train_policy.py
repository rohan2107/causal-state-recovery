import argparse
from pathlib import Path

from causalstate.agent.ppo import success_rate, train

RHO = 0.7
ENT_COEF = 0.05
EVAL_EVERY = 25_000
EVAL_EPISODES = 50


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--layout-seed", type=int, default=2)
    args = parser.parse_args()

    if args.layout_seed == 2:
        model_path = Path(
            f"models/ppo_{args.timesteps // 1000}k_rho07_seed{args.seed}.zip"
        )
    else:
        model_path = Path(
            f"models/ppo_{args.timesteps // 1000}k_rho07_seed{args.seed}"
            f"_layout{args.layout_seed}.zip"
        )

    model = train(
        total_timesteps=args.timesteps,
        seed=args.seed,
        layout_seed=args.layout_seed,
        rho=RHO,
        prune=False,
        save_path=model_path,
        ent_coef=ENT_COEF,
        eval_every=EVAL_EVERY,
        eval_episodes=EVAL_EPISODES,
    )

    final_success = success_rate(
        model,
        n_episodes=100,
        layout_seed=args.layout_seed,
        start_seed=123,
        rho=RHO,
        prune=False,
    )

    print("=== PPO training ===")
    print(f"timesteps: {args.timesteps}")
    print(f"seed: {args.seed}")
    print(f"layout_seed: {args.layout_seed}")
    print(f"rho: {RHO}")
    print(f"ent_coef: {ENT_COEF}")
    print(f"success_rate: {final_success:.3f}")
    print(f"model: {model_path}")


if __name__ == "__main__":
    main()
