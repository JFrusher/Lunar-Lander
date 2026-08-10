"""CLI entry point for the LunarLander control/RL lab.

Usage:
    lunar-lander run --controller [heuristic|rl]
    lunar-lander train --timesteps 100000
    lunar-lander benchmark --episodes 50
"""

import argparse
from typing import Optional

import gymnasium as gym

from .controllers import HeuristicController, RLAgent
from .utils.evaluation import SUCCESS_REWARD_THRESHOLD, run_benchmark
from .utils.pid_search import run_monte_carlo

DEFAULT_MODEL_NAME = "ppo_lunar_lander"


def build_controller(name: str, model_name: Optional[str] = None):
    if name == "heuristic":
        return HeuristicController()
    if name == "rl":
        agent = RLAgent()
        agent.load(model_name or DEFAULT_MODEL_NAME)
        return agent
    raise ValueError(f"Unknown controller: {name}")


def cmd_run(args: argparse.Namespace) -> None:
    controller = build_controller(args.controller, model_name=args.model)
    env = gym.make("LunarLander-v3", render_mode="human")

    episode = 0
    rewards = []
    try:
        while True:
            episode += 1
            observation, _ = env.reset(seed=args.seed)
            total_reward = 0.0
            terminated = truncated = False

            while not (terminated or truncated):
                action = controller.get_action(observation)
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)

            rewards.append(total_reward)
            successes = sum(r >= SUCCESS_REWARD_THRESHOLD for r in rewards)
            print(
                f"Episode {episode}: reward {total_reward:.1f}  "
                f"(avg {sum(rewards) / len(rewards):.1f}, landed {successes}/{episode})"
            )

            if not args.loop:
                break
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        env.close()


def cmd_train(args: argparse.Namespace) -> None:
    agent = RLAgent()
    saved_path = agent.train(total_timesteps=args.timesteps, save_path=DEFAULT_MODEL_NAME)
    print(f"Model saved to {saved_path}")


def cmd_pid_search(args: argparse.Namespace) -> None:
    run_monte_carlo(
        n_samples=args.samples,
        episodes_per_set=args.episodes,
        seed=args.seed,
        n_jobs=args.jobs,
        output_dir=args.output_dir,
    )


def cmd_benchmark(args: argparse.Namespace) -> None:
    controllers = {"Heuristic": HeuristicController()}

    rl_agent = RLAgent()
    try:
        rl_agent.load(DEFAULT_MODEL_NAME)
        controllers["RL (PPO)"] = rl_agent
    except FileNotFoundError:
        print(f"No trained model found ({DEFAULT_MODEL_NAME}); skipping RL controller. "
              f"Run 'python main.py train' first.")

    run_benchmark(controllers, num_episodes=args.episodes)


def main() -> None:
    parser = argparse.ArgumentParser(description="LunarLander control/RL lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Render a controller landing an episode")
    run_parser.add_argument("--controller", choices=["heuristic", "rl"], required=True)
    run_parser.add_argument("--seed", type=int, default=None)
    run_parser.add_argument(
        "--loop", action="store_true", help="Keep running episodes back-to-back until Ctrl+C"
    )
    run_parser.add_argument(
        "--model", default=None, help="Checkpoint name in models/ for --controller rl (default: ppo_lunar_lander)"
    )
    run_parser.set_defaults(func=cmd_run)

    train_parser = subparsers.add_parser("train", help="Train a PPO agent")
    train_parser.add_argument("--timesteps", type=int, default=100_000)
    train_parser.set_defaults(func=cmd_train)

    benchmark_parser = subparsers.add_parser("benchmark", help="Compare controllers")
    benchmark_parser.add_argument("--episodes", type=int, default=50)
    benchmark_parser.set_defaults(func=cmd_benchmark)

    pid_search_parser = subparsers.add_parser(
        "pid-search", help="Monte Carlo sweep of heuristic controller gains"
    )
    pid_search_parser.add_argument("--samples", type=int, default=200)
    pid_search_parser.add_argument("--episodes", type=int, default=30)
    pid_search_parser.add_argument("--seed", type=int, default=4316)
    pid_search_parser.add_argument(
        "--jobs", type=int, default=None, help="Parallel worker processes (default: CPU count)"
    )
    pid_search_parser.add_argument("--output-dir", default=None)
    pid_search_parser.set_defaults(func=cmd_pid_search)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
