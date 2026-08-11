"""Does throttle control beat bang-bang? Phase 8b of tmp/SPEED_ROADMAP.md.

Every RL result in this repo uses `Discrete(4)`: the main engine is full-on
or off, with nothing in between. Phase 1 measured the consequence -- while
firing, the engine nets only +0.158 velocity units/tick over gravity, and
there is no way to ask for less. Phase 6 then showed reward shaping has
saturated (4x the time penalty buys no speed) while the compressible part of
an episode still sits ~1.5x above a loose physical bound. Actuator
granularity is the leading remaining explanation.

**This runs continuous-PPO, not SAC.** The roadmap originally specified SAC,
but that changes the action space *and* the algorithm *and* the on/off-policy
family simultaneously, so a win could not be attributed to any of them.
Continuous-PPO changes exactly one thing against the existing baselines --
same algorithm, same validated 1M-timestep budget, same evaluation protocol.
It is both the cleaner experiment and the cheaper one.
"""

import multiprocessing
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .paths import new_run_dir
from .pid_search import HOLDOUT_SEED_START
from .time_penalty import CONTINUOUS_ENV_KWARGS, evaluate_controller_natural

# Phase 4 (old roadmap) measured 0.1 as the best flat penalty on discrete PPO
# and it remains the strongest single setting found. Both arms train under it
# so the comparison isolates the action space rather than re-testing shaping.
COMPARISON_PENALTY = 0.1

ARMS: Dict[str, Dict] = {
    "discrete": {},
    "continuous": CONTINUOUS_ENV_KWARGS,
}


def _train_and_eval_arm(args: tuple) -> Dict[str, float]:
    """Train one PPO model on one action space and score it. Runs in a worker."""
    arm, seed, total_timesteps, eval_episodes, penalty = args

    import torch

    # One thread per worker; torch's default of 4 oversubscribes the CPU
    # several times over once the pool is saturated (same as ppo_search).
    torch.set_num_threads(1)

    from ..controllers.rl_agent import RLAgent
    from .time_penalty import TimePenaltyWrapper

    env_kwargs = ARMS[arm]
    agent = RLAgent()
    saved_path = agent.train(
        total_timesteps=total_timesteps,
        save_path=f"ppo_{arm}_s{seed}",
        env_kwargs=env_kwargs,
        env_wrapper=(lambda env: TimePenaltyWrapper(env, penalty)) if penalty else None,
        hyperparams={"seed": seed, "verbose": 0},
    )
    metrics = evaluate_controller_natural(
        agent, eval_episodes, seed_start=HOLDOUT_SEED_START, env_kwargs=env_kwargs
    )
    return {"arm": arm, "seed": seed, "penalty": penalty, "detail": saved_path, **metrics}


def run_continuous_comparison(
    arms: Optional[List[str]] = None,
    seeds: Sequence[int] = (0, 100, 200),
    total_timesteps: int = 1_000_000,
    eval_episodes: int = 100,
    penalty: float = COMPARISON_PENALTY,
    n_jobs: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Train both action-space arms across `seeds` and report mean ± std.

    Runs 3 seeds from the start rather than a single-seed first pass: six
    1M-timestep runs fit one batch on 8 cores, so the multi-seed version is
    the same wall time, and single-seed results have failed their own repeat
    six times in this project.
    """
    arms = list(ARMS) if arms is None else arms
    out_dir = Path(output_dir) if output_dir else new_run_dir("continuous_compare")
    out_dir.mkdir(parents=True, exist_ok=True)

    work = [(a, s, total_timesteps, eval_episodes, penalty) for a in arms for s in seeds]
    n_jobs = min(n_jobs or os.cpu_count() or 1, len(work))
    print(f"Training {len(work)} PPO configs x {total_timesteps:,} timesteps "
          f"on {n_jobs} workers (penalty {penalty})...")

    rows = []
    with multiprocessing.Pool(processes=n_jobs) as pool:
        for i, r in enumerate(pool.imap_unordered(_train_and_eval_arm, work), 1):
            rows.append(r)
            print(f"  [{i}/{len(work)}] {r['arm']:10s} seed={r['seed']} "
                  f"reward={r['mean_reward']:.1f} success={r['success_rate_pct']:.0f}% "
                  f"flight={r['avg_flight_steps']:.1f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "continuous_compare_raw.csv", index=False)

    metrics = ["mean_reward", "success_rate_pct", "avg_landing_steps",
               "avg_flight_steps", "avg_settle_steps", "avg_touchdown_speed", "fuel_cost"]
    agg = df.groupby("arm")[metrics].agg(["mean", "std"])
    agg.to_csv(out_dir / "continuous_compare_aggregated.csv")

    print(f"\nRaw:        {out_dir / 'continuous_compare_raw.csv'}")
    print(f"Aggregated: {out_dir / 'continuous_compare_aggregated.csv'}")
    print(agg.to_string())
    return df
