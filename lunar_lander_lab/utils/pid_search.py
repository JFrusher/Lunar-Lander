"""Monte Carlo sweep of HeuristicController gains.

Latin-Hypercube-samples the gain space, evaluates each sample over many
episodes (multiprocessed across CPU cores), and writes a dataset + plots
for offline analysis/optimization.
"""

import json
import multiprocessing
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.evaluation import CRASH_REWARD_THRESHOLD, SUCCESS_REWARD_THRESHOLD

# Bounds for the four gains that most directly shape the descent trajectory.
CORE_PARAM_SPACE: Dict[str, Tuple[float, float]] = {
    "ANGLE_GAIN_POS": (0.1, 1.0),
    "ANGLE_GAIN_VEL": (0.2, 2.0),
    "DESCENT_GAIN": (0.1, 1.5),
    "TARGET_DESCENT_SPEED": (-0.40, -0.05),
}


def _latin_hypercube_unit(n_samples: int, n_dims: int, rng: np.random.Generator) -> np.ndarray:
    """Stratified samples in [0, 1)^n_dims: one random draw per bin per dim."""
    result = np.empty((n_samples, n_dims))
    for d in range(n_dims):
        bin_edges = rng.permutation(n_samples)
        result[:, d] = (bin_edges + rng.random(n_samples)) / n_samples
    return result


def sample_gain_sets(
    n_samples: int, param_space: Dict[str, Tuple[float, float]], seed: int = 0
) -> List[Dict[str, float]]:
    rng = np.random.default_rng(seed)
    names = list(param_space.keys())
    unit = _latin_hypercube_unit(n_samples, len(names), rng)

    gain_sets = []
    for row in unit:
        gains = {}
        for i, name in enumerate(names):
            low, high = param_space[name]
            gains[name] = float(low + row[i] * (high - low))
        gain_sets.append(gains)
    return gain_sets


def _evaluate_gain_set(args: Tuple[Dict[str, float], int, str]) -> Dict[str, float]:
    """Run one gain set for `episodes` seeded episodes. Runs in a worker process."""
    gains, episodes, env_name = args

    import gymnasium as gym

    from controllers.heuristic import HeuristicController

    controller = HeuristicController()
    for name, value in gains.items():
        setattr(controller, name, value)

    env = gym.make(env_name)
    rewards, steps_taken = [], []
    successes = crashes = 0

    for seed in range(episodes):
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

    env.close()

    result = dict(gains)
    result.update(
        {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "success_rate_pct": 100 * successes / episodes,
            "crash_rate_pct": 100 * crashes / episodes,
            "avg_steps": float(np.mean(steps_taken)),
        }
    )
    return result


def _plot_param_scatter(df: pd.DataFrame, param_names: List[str], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, name in zip(axes.flat, param_names):
        scatter = ax.scatter(
            df[name], df["mean_reward"], c=df["success_rate_pct"], cmap="viridis", s=18
        )
        ax.set_xlabel(name)
        ax.set_ylabel("mean_reward")
    fig.colorbar(scatter, ax=axes, label="success_rate_pct", shrink=0.8)
    fig.suptitle("Gain Value vs. Mean Reward (color = success rate)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_correlation_heatmap(df: pd.DataFrame, path: Path) -> None:
    columns = [c for c in df.columns if c != "std_reward"]
    corr = df[columns].corr()

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)), columns, rotation=45, ha="right")
    ax.set_yticks(range(len(columns)), columns)
    for i in range(len(columns)):
        for j in range(len(columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="correlation")
    fig.suptitle("Parameter / Metric Correlation")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_monte_carlo(
    n_samples: int = 200,
    episodes_per_set: int = 30,
    seed: int = 0,
    env_name: str = "LunarLander-v3",
    param_space: Optional[Dict[str, Tuple[float, float]]] = None,
    output_dir: str = "pid_search_results",
    n_jobs: Optional[int] = None,
) -> pd.DataFrame:
    """Latin-Hypercube-sample `param_space`, evaluate each set, save dataset + plots."""
    param_space = param_space or CORE_PARAM_SPACE
    gain_sets = sample_gain_sets(n_samples, param_space, seed=seed)
    work_items = [(gains, episodes_per_set, env_name) for gains in gain_sets]

    n_jobs = n_jobs or os.cpu_count() or 1
    results = []
    print(f"Evaluating {n_samples} gain sets x {episodes_per_set} episodes on {n_jobs} workers...")
    with multiprocessing.Pool(processes=n_jobs) as pool:
        for i, result in enumerate(pool.imap_unordered(_evaluate_gain_set, work_items), 1):
            results.append(result)
            if i % max(1, n_samples // 10) == 0 or i == n_samples:
                print(f"  [{i}/{n_samples}] evaluated")

    df = pd.DataFrame(results)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "pid_search_results.csv"
    df.to_csv(csv_path, index=False)

    param_names = list(param_space.keys())
    _plot_param_scatter(df, param_names, out_dir / "pid_search_scatter.png")
    _plot_correlation_heatmap(df, out_dir / "pid_search_correlation.png")

    best_row = df.sort_values(["mean_reward", "success_rate_pct"], ascending=False).iloc[0]
    summary = {
        "best_gains": {name: float(best_row[name]) for name in param_names},
        "mean_reward": float(best_row["mean_reward"]),
        "success_rate_pct": float(best_row["success_rate_pct"]),
        "crash_rate_pct": float(best_row["crash_rate_pct"]),
        "avg_steps": float(best_row["avg_steps"]),
    }
    with open(out_dir / "best_gains.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDataset: {csv_path}")
    print(f"Plots:   {out_dir / 'pid_search_scatter.png'}, {out_dir / 'pid_search_correlation.png'}")
    print("Best gain set found:")
    print(json.dumps(summary, indent=2))
    return df
