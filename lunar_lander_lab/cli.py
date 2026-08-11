"""CLI entry point for the LunarLander control/RL lab.

Usage:
    lunar-lander run --controller [heuristic|rl]
    lunar-lander train --timesteps 100000
    lunar-lander benchmark --episodes 50
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import gymnasium as gym

from .controllers import HeuristicController, RLAgent
from .utils.continuous_compare import ARMS, COMPARISON_PENALTY, run_continuous_comparison
from .utils.evaluation import SUCCESS_REWARD_THRESHOLD, run_benchmark
from .utils.lockout_sweep import run_lockout_sweep
from .utils.penalty_curriculum import SCHEDULES, TARGET_PENALTY, run_penalty_curriculum_sweep
from .utils.pid_search import CORE_PARAM_SPACE, EXTENDED_PARAM_SPACE, run_monte_carlo
from .utils.ppo_convergence import run_ppo_convergence_check
from .utils.ppo_search import DEFAULT_TIMESTEPS, run_ppo_search
from .utils.speed_ceiling import run_speed_ceiling
from .utils.time_penalty import EngineLockoutWrapper, run_multi_seed_sweep, run_time_penalty_sweep

DEFAULT_MODEL_NAME = "ppo_lunar_lander"


def build_controller(
    name: str, model_name: Optional[str] = None, gains_path: Optional[str] = None
):
    if name == "heuristic":
        controller = HeuristicController()
        if gains_path:
            # Override the shipped gains so an older set can be flown
            # side-by-side with the current one. Same setattr mechanism
            # pid_search uses to evaluate a sampled gain set.
            for gain, value in json.loads(Path(gains_path).read_text()).items():
                if not hasattr(controller, gain):
                    raise ValueError(f"{gains_path}: unknown gain {gain!r}")
                setattr(controller, gain, value)
        return controller
    if name == "rl":
        agent = RLAgent()
        agent.load(model_name or DEFAULT_MODEL_NAME)
        return agent
    raise ValueError(f"Unknown controller: {name}")


def cmd_run(args: argparse.Namespace) -> None:
    controller = build_controller(
        args.controller, model_name=args.model, gains_path=args.gains
    )
    env = gym.make("LunarLander-v3", render_mode="human")
    if args.lockout_steps is not None or args.lockout_altitude is not None:
        env = EngineLockoutWrapper(
            env, lockout_steps=args.lockout_steps, altitude_threshold=args.lockout_altitude
        )

    episode = 0
    rewards: list[float] = []
    landed_flight: list[int] = []
    landed_settle: list[int] = []
    try:
        while True:
            episode += 1
            observation, _ = env.reset(seed=args.seed)
            total_reward = 0.0
            steps = 0
            first_contact = None
            terminated = truncated = False

            while not (terminated or truncated):
                had_contact = bool(observation[6] or observation[7])
                action = controller.get_action(observation)
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                steps += 1
                if first_contact is None and not had_contact and (observation[6] or observation[7]):
                    first_contact = steps

            rewards.append(total_reward)
            landed = total_reward >= SUCCESS_REWARD_THRESHOLD
            # Speed is the whole point of this controller work, so the loop
            # view reports it -- split at first leg contact, because
            # everything after that is Box2D settling the lander to sleep and
            # no controller can fly its way out of it (SPEED_ROADMAP Phase 6).
            if landed and first_contact is not None:
                landed_flight.append(first_contact)
                landed_settle.append(steps - first_contact)

            successes = len(landed_flight)
            line = (
                f"Episode {episode}: {'LANDED' if landed else 'failed'}  "
                f"reward {total_reward:.1f}  steps {steps}"
            )
            if first_contact is not None:
                line += f" ({first_contact} flight + {steps - first_contact} settling)"
            print(line)
            if successes:
                print(
                    f"    running: {successes}/{episode} landed, "
                    f"avg {sum(landed_flight) / successes:.0f} flight "
                    f"+ {sum(landed_settle) / successes:.0f} settling "
                    f"= {(sum(landed_flight) + sum(landed_settle)) / successes:.0f} steps, "
                    f"avg reward {sum(rewards) / len(rewards):.1f}"
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
        param_space=CORE_PARAM_SPACE if args.param_space == "core" else EXTENDED_PARAM_SPACE,
        time_penalty=args.time_penalty,
        n_jobs=args.jobs,
        output_dir=args.output_dir,
    )


def cmd_time_penalty_sweep(args: argparse.Namespace) -> None:
    run_time_penalty_sweep(
        rl_timesteps=args.rl_timesteps,
        pid_samples=args.pid_samples,
        pid_episodes=args.pid_episodes,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        n_jobs=args.jobs,
    )


def cmd_multi_seed_sweep(args: argparse.Namespace) -> None:
    run_multi_seed_sweep(
        seeds=args.seeds,
        rl_timesteps=args.rl_timesteps,
        pid_samples=args.pid_samples,
        pid_episodes=args.pid_episodes,
        eval_episodes=args.eval_episodes,
        n_jobs=args.jobs,
    )


def cmd_ppo_convergence_check(args: argparse.Namespace) -> None:
    run_ppo_convergence_check(
        eval_episodes=args.eval_episodes,
        seed=args.seed,
    )


def cmd_hparam_search(args: argparse.Namespace) -> None:
    run_ppo_search(
        n_samples=args.samples,
        total_timesteps=args.timesteps,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        n_jobs=args.jobs,
    )


def cmd_speed_ceiling(args: argparse.Namespace) -> None:
    run_speed_ceiling(n_samples=args.samples, seed_start=args.seed)


def cmd_lockout_sweep(args: argparse.Namespace) -> None:
    run_lockout_sweep(
        total_timesteps=args.rl_timesteps,
        eval_episodes=args.eval_episodes,
        seeds=args.seeds,
        n_jobs=args.jobs,
    )


def cmd_penalty_curriculum_sweep(args: argparse.Namespace) -> None:
    run_penalty_curriculum_sweep(
        shapes=args.shapes,
        seeds=args.seeds,
        total_timesteps=args.rl_timesteps,
        eval_episodes=args.eval_episodes,
        target=args.target,
        n_jobs=args.jobs,
    )


def cmd_continuous_compare(args: argparse.Namespace) -> None:
    run_continuous_comparison(
        arms=args.arms,
        seeds=args.seeds,
        total_timesteps=args.rl_timesteps,
        eval_episodes=args.eval_episodes,
        penalty=args.penalty,
        n_jobs=args.jobs,
    )


def cmd_benchmark(args: argparse.Namespace) -> None:
    controllers = {"Heuristic": HeuristicController()}

    rl_agent = RLAgent()
    try:
        rl_agent.load(DEFAULT_MODEL_NAME)
        controllers["RL (PPO)"] = rl_agent
    except FileNotFoundError:
        print(f"No trained model found ({DEFAULT_MODEL_NAME}); skipping RL controller. "
              f"Run 'lunar-lander train' first.")

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
    run_parser.add_argument(
        "--lockout-steps", type=int, default=None,
        help="Block the main engine for the first N steps (tmp/SPEED_ROADMAP.md Phase 2)",
    )
    run_parser.add_argument(
        "--lockout-altitude", type=float, default=None,
        help="Block the main engine while normalized altitude is above this",
    )
    run_parser.add_argument(
        "--gains", default=None,
        help="JSON gains file overriding configs/heuristic_gains.json, for flying an "
             "older gain set side-by-side with the current one",
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
    pid_search_parser.add_argument(
        "--param-space", choices=["core", "extended"], default="core",
        help="core = the 5 swept gains; extended = 8D, adding the 3 attitude gains",
    )
    pid_search_parser.add_argument(
        "--time-penalty", type=float, default=0.0,
        help="Per-step penalty applied when ranking gain sets (not during evaluation)",
    )
    pid_search_parser.add_argument("--output-dir", default=None)
    pid_search_parser.set_defaults(func=cmd_pid_search)

    time_penalty_parser = subparsers.add_parser(
        "time-penalty-sweep",
        help="Sweep a per-step time penalty across both controllers and plot the trade-off",
    )
    time_penalty_parser.add_argument("--rl-timesteps", type=int, default=150_000)
    time_penalty_parser.add_argument("--pid-samples", type=int, default=200)
    time_penalty_parser.add_argument("--pid-episodes", type=int, default=30)
    time_penalty_parser.add_argument("--eval-episodes", type=int, default=30)
    time_penalty_parser.add_argument("--seed", type=int, default=0)
    time_penalty_parser.add_argument(
        "--jobs", type=int, default=None, help="Parallel worker processes for pid-search (default: CPU count)"
    )
    time_penalty_parser.set_defaults(func=cmd_time_penalty_sweep)

    multi_seed_parser = subparsers.add_parser(
        "multi-seed-sweep",
        help="Time-penalty sweep repeated across seeds, aggregated to mean ± std",
    )
    multi_seed_parser.add_argument("--seeds", type=int, nargs="+", default=[0, 100, 200])
    multi_seed_parser.add_argument("--rl-timesteps", type=int, default=1_000_000)
    multi_seed_parser.add_argument("--pid-samples", type=int, default=200)
    multi_seed_parser.add_argument("--pid-episodes", type=int, default=30)
    multi_seed_parser.add_argument("--eval-episodes", type=int, default=100)
    multi_seed_parser.add_argument("--jobs", type=int, default=None)
    multi_seed_parser.set_defaults(func=cmd_multi_seed_sweep)

    ppo_convergence_parser = subparsers.add_parser(
        "ppo-convergence-check",
        help="Train PPO at increasing timestep budgets and plot reward/success vs. timesteps",
    )
    ppo_convergence_parser.add_argument("--eval-episodes", type=int, default=30)
    ppo_convergence_parser.add_argument("--seed", type=int, default=0)
    ppo_convergence_parser.set_defaults(func=cmd_ppo_convergence_check)

    hparam_parser = subparsers.add_parser(
        "hparam-search", help="Monte Carlo sweep of PPO hyperparameters"
    )
    hparam_parser.add_argument("--samples", type=int, default=12)
    hparam_parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    hparam_parser.add_argument("--eval-episodes", type=int, default=30)
    hparam_parser.add_argument("--seed", type=int, default=0)
    hparam_parser.add_argument(
        "--jobs", type=int, default=None, help="Parallel worker processes (default: CPU count)"
    )
    hparam_parser.set_defaults(func=cmd_hparam_search)

    speed_ceiling_parser = subparsers.add_parser(
        "speed-ceiling",
        help="Idealized point-mass minimum-time descent bound (tmp/SPEED_ROADMAP.md Phase 1)",
    )
    speed_ceiling_parser.add_argument(
        "--samples", type=int, default=20,
        help="Reset samples used to measure engine/gravity constants and the start-state distribution",
    )
    speed_ceiling_parser.add_argument("--seed", type=int, default=0)
    speed_ceiling_parser.set_defaults(func=cmd_speed_ceiling)

    lockout_parser = subparsers.add_parser(
        "lockout-sweep",
        help="Sweep engine-lockout gating (step-based and altitude-based) across both controllers "
        "(tmp/SPEED_ROADMAP.md Phase 2)",
    )
    lockout_parser.add_argument("--rl-timesteps", type=int, default=1_000_000)
    lockout_parser.add_argument("--eval-episodes", type=int, default=100)
    lockout_parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0],
        help="RL seeds -- default is a single-seed first pass; pass 3 to confirm a winner",
    )
    lockout_parser.add_argument(
        "--jobs", type=int, default=None, help="Parallel worker processes (default: CPU count)"
    )
    lockout_parser.set_defaults(func=cmd_lockout_sweep)

    curriculum_parser = subparsers.add_parser(
        "penalty-curriculum-sweep",
        help="Anneal the time penalty over training instead of fixing it "
        "(tmp/SPEED_ROADMAP.md Phase 5)",
    )
    curriculum_parser.add_argument(
        "--shapes", nargs="+", choices=sorted(SCHEDULES), default=sorted(SCHEDULES)
    )
    curriculum_parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    curriculum_parser.add_argument("--rl-timesteps", type=int, default=1_000_000)
    curriculum_parser.add_argument("--eval-episodes", type=int, default=100)
    curriculum_parser.add_argument(
        "--target", type=float, default=TARGET_PENALTY,
        help="Penalty the schedule ramps up to (default: the level a flat penalty collapses at)",
    )
    curriculum_parser.add_argument("--jobs", type=int, default=None)
    curriculum_parser.set_defaults(func=cmd_penalty_curriculum_sweep)

    continuous_parser = subparsers.add_parser(
        "continuous-compare",
        help="Discrete vs continuous action space at matched algorithm/budget/seeds "
        "(tmp/SPEED_ROADMAP.md Phase 8b)",
    )
    continuous_parser.add_argument(
        "--arms", nargs="+", choices=sorted(ARMS), default=sorted(ARMS)
    )
    continuous_parser.add_argument("--seeds", type=int, nargs="+", default=[0, 100, 200])
    continuous_parser.add_argument("--rl-timesteps", type=int, default=1_000_000)
    continuous_parser.add_argument("--eval-episodes", type=int, default=100)
    continuous_parser.add_argument(
        "--penalty", type=float, default=COMPARISON_PENALTY,
        help="Time penalty both arms train under (default: the best flat level measured)",
    )
    continuous_parser.add_argument("--jobs", type=int, default=None)
    continuous_parser.set_defaults(func=cmd_continuous_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
