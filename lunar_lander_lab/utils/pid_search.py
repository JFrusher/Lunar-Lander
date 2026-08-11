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

from .evaluation import CRASH_REWARD_THRESHOLD, SUCCESS_REWARD_THRESHOLD
from .paths import new_run_dir

# Bounds for the gains that matter, per sweep-v1 analysis:
# - ANGLE_GAIN_POS dropped: corr with reward was -0.04 in v1, fixed at its default.
# - TARGET_DESCENT_SPEED widened: v1 best cluster sat at the -0.05 edge of the old range.
# - DESCENT_GAIN widened: v1 top result (1.32) sat close to the old 1.5 ceiling.
# - ANGLE_THRESHOLD / HOVER_THRESHOLD added: gate action selection directly, untested in v1.
CORE_PARAM_SPACE: Dict[str, Tuple[float, float]] = {
    "ANGLE_GAIN_VEL": (0.2, 2.0),
    "DESCENT_GAIN": (0.1, 2.0),
    "TARGET_DESCENT_SPEED": (-0.40, -0.02),
    "ANGLE_THRESHOLD": (0.01, 0.2),
    "HOVER_THRESHOLD": (0.01, 0.2),
}

# The three attitude gains sweep-v1 left hardcoded, bracketing their original
# defaults (0.5 / 0.5 / 1.0) in both directions. ANGLE_GAIN_POS was dropped from
# v1 on a -0.04 reward correlation, which is weak evidence for "doesn't matter"
# when it was never varied jointly with the attitude gains it interacts with.
EXTENDED_PARAM_SPACE: Dict[str, Tuple[float, float]] = {
    **CORE_PARAM_SPACE,
    "ANGLE_GAIN_POS": (0.1, 1.5),
    "ANGLE_ERROR_GAIN": (0.1, 1.5),
    "ANGULAR_VEL_GAIN": (0.2, 2.0),
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


# Held-out episodes for re-scoring the search's top-k. Far clear of the
# episode seeds the search itself ranks on (0..episodes_per_set-1), so a
# winner can't be validated against the episodes that selected it.
HOLDOUT_SEED_START = 10_000


def select_fastest_within_floor(
    df: pd.DataFrame, success_floor: float, top_k: int = 10
) -> pd.DataFrame:
    """Rank gain sets by landing speed subject to a success-rate floor.

    A different objective from `run_monte_carlo`'s penalized argmax, which
    maximizes `mean_reward - penalty * avg_steps`. That blend lets a gain set
    trade success away for speed at whatever exchange rate the penalty
    implies; this states the constraint directly instead -- be at least this
    reliable, then be as fast as possible.

    Rows that never landed (`avg_steps_success` is NaN) are dropped rather
    than sorted, or they would rank as infinitely fast.
    """
    eligible = df[
        (df["success_rate_pct"] >= success_floor) & df["avg_steps_success"].notna()
    ]
    return eligible.nsmallest(top_k, "avg_steps_success")


def _evaluate_gain_set(args: Tuple[Dict[str, float], int, str, int]) -> Dict[str, float]:
    """Run one gain set for `episodes` seeded episodes. Runs in a worker process."""
    gains, episodes, env_name, seed_start = args

    import gymnasium as gym

    from ..controllers.heuristic import HeuristicController

    controller = HeuristicController()
    for name, value in gains.items():
        setattr(controller, name, value)

    env = gym.make(env_name)
    rewards, steps_taken, success_steps = [], [], []
    successes = crashes = timeouts = 0

    for seed in range(seed_start, seed_start + episodes):
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
            success_steps.append(steps)
        elif terminated and total_reward <= -CRASH_REWARD_THRESHOLD:
            crashes += 1
        if truncated:
            timeouts += 1

    env.close()

    result = dict(gains)
    result.update(
        {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "success_rate_pct": 100 * successes / episodes,
            "crash_rate_pct": 100 * crashes / episodes,
            "timeout_rate_pct": 100 * timeouts / episodes,
            "avg_steps": float(np.mean(steps_taken)),
            "avg_steps_success": float(np.mean(success_steps)) if success_steps else float("nan"),
        }
    )
    return result


def _plot_param_scatter(df: pd.DataFrame, param_names: List[str], path: Path) -> None:
    cols = min(3, len(param_names))
    rows = -(-len(param_names) // cols)  # ceil
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4 * rows), squeeze=False)

    scatter = None
    for i, name in enumerate(param_names):
        ax = axes.flat[i]
        scatter = ax.scatter(
            df[name], df["mean_reward"], c=df["success_rate_pct"], cmap="viridis", s=18
        )
        ax.set_xlabel(name)
        ax.set_ylabel("mean_reward")
    for ax in axes.flat[len(param_names):]:
        ax.set_visible(False)

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
    output_dir: Optional[str] = None,
    n_jobs: Optional[int] = None,
    time_penalty: float = 0.0,
    holdout_top_k: int = 10,
    holdout_episodes: int = 100,
) -> pd.DataFrame:
    """Latin-Hypercube-sample `param_space`, evaluate each set, save dataset + plots.

    The reported winner is chosen by re-scoring the top `holdout_top_k` on
    `holdout_episodes` episodes the search never ranked against — see the
    comment at the selection step for why.
    """
    param_space = param_space or CORE_PARAM_SPACE
    gain_sets = sample_gain_sets(n_samples, param_space, seed=seed)
    work_items = [(gains, episodes_per_set, env_name, 0) for gains in gain_sets]

    n_jobs = n_jobs or os.cpu_count() or 1
    results = []
    print(f"Evaluating {n_samples} gain sets x {episodes_per_set} episodes on {n_jobs} workers...")
    with multiprocessing.Pool(processes=n_jobs) as pool:
        for i, result in enumerate(pool.imap_unordered(_evaluate_gain_set, work_items), 1):
            results.append(result)
            if i % max(1, n_samples // 10) == 0 or i == n_samples:
                print(f"  [{i}/{n_samples}] evaluated")

    df = pd.DataFrame(results)
    out_dir = Path(output_dir) if output_dir else new_run_dir("pid_search")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "pid_search_results.csv"
    df.to_csv(csv_path, index=False)

    with open(out_dir / "sweep_manifest.json", "w") as f:
        json.dump(
            {
                "env_name": env_name,
                "n_samples": n_samples,
                "episodes_per_set": episodes_per_set,
                "seed": seed,
                "param_space": param_space,
                "time_penalty": time_penalty,
                "holdout_top_k": holdout_top_k,
                "holdout_episodes": holdout_episodes,
                "holdout_seed_start": HOLDOUT_SEED_START,
            },
            f,
            indent=2,
        )

    param_names = list(param_space.keys())
    _plot_param_scatter(df, param_names, out_dir / "pid_search_scatter.png")
    _plot_correlation_heatmap(df, out_dir / "pid_search_correlation.png")

    penalized_score = df["mean_reward"] - time_penalty * df["avg_steps"]
    search_best = df.loc[penalized_score.idxmax()]

    # Ranking each gain set on the same episodes used to pick the winner
    # inflates the winner's score, and the inflation grows with n_samples
    # (measured at +6 reward for n=125 up to +31 for n=500). Re-score the
    # top-k on episodes the search never saw, and let those decide.
    k = min(holdout_top_k, len(df))
    top_k = df.loc[penalized_score.nlargest(k).index]
    holdout_items = [
        ({name: float(row[name]) for name in param_names},
         holdout_episodes, env_name, HOLDOUT_SEED_START)
        for _, row in top_k.iterrows()
    ]
    print(f"Re-scoring top {k} on {holdout_episodes} held-out episodes...")
    with multiprocessing.Pool(processes=min(n_jobs, k)) as pool:
        holdout_df = pd.DataFrame(pool.map(_evaluate_gain_set, holdout_items))

    holdout_df.to_csv(out_dir / "holdout_top_k.csv", index=False)
    holdout_score = holdout_df["mean_reward"] - time_penalty * holdout_df["avg_steps"]
    best_row = holdout_df.loc[holdout_score.idxmax()]

    summary = {
        "best_gains": {name: float(best_row[name]) for name in param_names},
        # Metrics below are held-out: measured on episodes the search never
        # ranked against. Use these when comparing across runs.
        "mean_reward": float(best_row["mean_reward"]),
        "success_rate_pct": float(best_row["success_rate_pct"]),
        "crash_rate_pct": float(best_row["crash_rate_pct"]),
        "avg_steps": float(best_row["avg_steps"]),
        "avg_steps_success": float(best_row["avg_steps_success"]),
        "time_penalty": time_penalty,
        "holdout_episodes": holdout_episodes,
        "holdout_top_k": k,
        # What the old (search-set-only) selection would have reported, kept
        # so the optimism is visible rather than silently corrected away.
        "search_set_best_mean_reward": float(search_best["mean_reward"]),
        "search_set_optimism": float(search_best["mean_reward"] - best_row["mean_reward"]),
    }
    with open(out_dir / "best_gains.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDataset: {csv_path}")
    print(f"Plots:   {out_dir / 'pid_search_scatter.png'}, {out_dir / 'pid_search_correlation.png'}")
    print("Best gain set found:")
    print(json.dumps(summary, indent=2))
    return df
