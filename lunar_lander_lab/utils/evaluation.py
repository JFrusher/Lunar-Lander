"""Benchmarking utilities: run controllers head-to-head and compare metrics."""

from typing import Dict

import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd

from controllers.base import BaseController

# A landing is considered successful once the episode's total reward crosses
# this threshold (LunarLander awards +100 for a safe landing on top of the
# shaping reward accrued during descent).
SUCCESS_REWARD_THRESHOLD = 200
# A crash is signalled by the large one-off penalty LunarLander applies when
# the lander body collides with the ground/moon.
CRASH_REWARD_THRESHOLD = 100


def run_benchmark(
    controllers: Dict[str, BaseController],
    env_name: str = "LunarLander-v3",
    num_episodes: int = 50,
    plot_path: str = "benchmark_results.png",
) -> pd.DataFrame:
    """Evaluate each controller over the same set of episode seeds.

    Returns a DataFrame of per-controller summary metrics and writes a
    comparison bar chart to ``plot_path``.
    """
    seeds = list(range(num_episodes))
    env = gym.make(env_name)

    rows = []
    for name, controller in controllers.items():
        rewards = []
        steps_taken = []
        successes = 0
        crashes = 0

        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            total_reward = 0.0
            steps = 0
            terminated = truncated = False

            while not (terminated or truncated):
                action = controller.get_action(observation)
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1

            rewards.append(total_reward)
            steps_taken.append(steps)
            if total_reward >= SUCCESS_REWARD_THRESHOLD:
                successes += 1
            elif terminated and total_reward <= -CRASH_REWARD_THRESHOLD:
                crashes += 1

        rows.append(
            {
                "controller": name,
                "mean_reward": sum(rewards) / len(rewards),
                "success_rate_pct": 100 * successes / num_episodes,
                "avg_steps": sum(steps_taken) / len(steps_taken),
                "crash_rate_pct": 100 * crashes / num_episodes,
            }
        )

    env.close()
    results = pd.DataFrame(rows).set_index("controller")
    print(results.to_string())
    _plot_results(results, plot_path)
    return results


def _plot_results(results: pd.DataFrame, plot_path: str) -> None:
    metrics = [
        ("mean_reward", "Mean Reward"),
        ("success_rate_pct", "Success / Landing Rate (%)"),
        ("avg_steps", "Avg Flight Time (steps)"),
        ("crash_rate_pct", "Crash Rate (%)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (column, title) in zip(axes.flat, metrics):
        results[column].plot(kind="bar", ax=ax, color="steelblue")
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)

    fig.suptitle("Controller Benchmark Comparison")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
