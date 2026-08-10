"""Monte Carlo sweep of PPO hyperparameters.

Same shape as `pid_search.py` — Latin-Hypercube-samples the hyperparameter
space, trains one PPO model per sample at a fixed seed and timestep budget,
evaluates each with natural (unpenalized) reward, and writes a ranked
dataset for offline analysis.

`DEFAULT_HYPERPARAMS` in `rl_agent.py` is a comment-labeled guess that has
never been validated; Phase 1's convergence check showed it needs ~1M
timesteps just to reach 100% success, so "converges faster" is part of what
this sweep is measuring, not only "scores higher".
"""

import json
import multiprocessing
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .paths import new_run_dir
from .pid_search import sample_gain_sets

# `learning_rate` is sampled log-uniformly (as log10) — a linear sweep of
# 1e-4..1e-3 would spend 90% of its samples above 5e-4.
#
# `gamma` is sampled as log10(1 - gamma) for the same reason, and because the
# first sweep bounded it at (0.99, 0.999) with the default sitting exactly on
# the upper edge — so the search structurally could not test whether the
# default was at a boundary optimum. log10(1-gamma) in (-4, -2) spans
# gamma 0.99 .. 0.9999, putting the old 0.999 default mid-range.
PPO_PARAM_SPACE: Dict[str, Tuple[float, float]] = {
    "learning_rate_log10": (-4.0, -3.0),
    "n_steps": (512, 4096),
    "ent_coef": (0.0, 0.05),
    "one_minus_gamma_log10": (-4.0, -2.0),
    "gae_lambda": (0.9, 0.99),
}

# Phase 1's measured floor: PPO scores 0% success at 200k and only reaches
# 100% at 700k-1M. Anything less confounds "bad hyperparameters" with
# "not trained yet".
DEFAULT_TIMESTEPS = 1_000_000

# SB3 warns when n_steps isn't a whole number of minibatches.
_BATCH_SIZE = 64


def to_hyperparams(sample: Dict[str, float]) -> Dict[str, float]:
    """Convert one raw Latin-Hypercube sample into real PPO keyword arguments."""
    return {
        "learning_rate": 10 ** sample["learning_rate_log10"],
        "n_steps": max(_BATCH_SIZE, round(sample["n_steps"] / _BATCH_SIZE) * _BATCH_SIZE),
        "ent_coef": sample["ent_coef"],
        "gamma": 1 - 10 ** sample["one_minus_gamma_log10"],
        "gae_lambda": sample["gae_lambda"],
    }


def _evaluate_hparam_set(args: Tuple[int, Dict[str, float], int, int, int]) -> Dict[str, float]:
    """Train + evaluate one hyperparameter set. Runs in a worker process."""
    index, sample, total_timesteps, eval_episodes, seed = args

    import torch

    # One thread per worker: torch defaults to 4, which would oversubscribe
    # the CPU several times over once the Pool is saturated.
    torch.set_num_threads(1)

    from ..controllers.rl_agent import RLAgent
    from .time_penalty import evaluate_controller_natural

    hyperparams = to_hyperparams(sample)
    agent = RLAgent()
    saved_path = agent.train(
        total_timesteps=total_timesteps,
        save_path=f"ppo_hparam_{index}",
        hyperparams={**hyperparams, "seed": seed, "verbose": 0},
    )
    metrics = evaluate_controller_natural(agent, eval_episodes, seed_start=seed)
    return {**hyperparams, "detail": saved_path, **metrics}


def run_ppo_search(
    n_samples: int = 12,
    total_timesteps: int = DEFAULT_TIMESTEPS,
    eval_episodes: int = 30,
    seed: int = 0,
    param_space: Optional[Dict[str, Tuple[float, float]]] = None,
    output_dir: Optional[str] = None,
    n_jobs: Optional[int] = None,
) -> pd.DataFrame:
    """Latin-Hypercube-sample `param_space`, train/evaluate each, rank by mean_reward."""
    param_space = param_space or PPO_PARAM_SPACE
    samples = sample_gain_sets(n_samples, param_space, seed=seed)
    work_items = [
        (i, sample, total_timesteps, eval_episodes, seed) for i, sample in enumerate(samples)
    ]

    n_jobs = n_jobs or os.cpu_count() or 1
    results: List[Dict[str, float]] = []
    print(
        f"Training {n_samples} PPO configs x {total_timesteps:,} timesteps "
        f"on {n_jobs} workers..."
    )
    with multiprocessing.Pool(processes=n_jobs) as pool:
        for i, result in enumerate(pool.imap_unordered(_evaluate_hparam_set, work_items), 1):
            results.append(result)
            print(f"  [{i}/{n_samples}] mean_reward={result['mean_reward']:.1f}")

    df = pd.DataFrame(results).sort_values("mean_reward", ascending=False)
    out_dir = Path(output_dir) if output_dir else new_run_dir("ppo_search")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "ppo_search_results.csv"
    df.to_csv(csv_path, index=False)

    param_names = list(to_hyperparams(samples[0]))
    best_row = df.iloc[0]
    summary = {
        "best_hyperparams": {name: float(best_row[name]) for name in param_names},
        "mean_reward": float(best_row["mean_reward"]),
        "success_rate_pct": float(best_row["success_rate_pct"]),
        "crash_rate_pct": float(best_row["crash_rate_pct"]),
        "avg_landing_steps": float(best_row["avg_landing_steps"]),
        "total_timesteps": total_timesteps,
        "n_samples": n_samples,
        "seed": seed,
    }
    with open(out_dir / "best_hyperparams.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDataset: {csv_path}")
    print("Best hyperparameter set found:")
    print(json.dumps(summary, indent=2))
    return df


if __name__ == "__main__":
    samples = sample_gain_sets(20, PPO_PARAM_SPACE, seed=0)
    for sample in samples:
        hp = to_hyperparams(sample)
        assert 1e-4 <= hp["learning_rate"] <= 1e-3, hp
        assert hp["n_steps"] % _BATCH_SIZE == 0, hp
        assert 0.99 <= hp["gamma"] <= 0.9999, hp
    print("ppo_search self-check OK")
