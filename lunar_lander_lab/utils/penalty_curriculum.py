"""Anneal the time penalty over training instead of fixing it at step 0.

Old roadmap Phase 4 measured a flat per-step penalty across 6 levels: 0.1 is
free (22% faster landings at no cost), 0.4 collapses the policy to 13.7%
success. Nothing explained the cliff between them.

One hypothesis: at a flat 0.4 the penalty is already shaping the reward
before the policy can land at all, so it never discovers landing in the
first place. If that's the mechanism, a policy that learns to land under a
mild penalty and is then squeezed harder should survive further up. This
module tests that. Phase 5 of tmp/SPEED_ROADMAP.md.
"""

import multiprocessing
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import gymnasium as gym
import pandas as pd

from .evaluation import evaluate_controller_natural
from .paths import new_run_dir
from .pid_search import HOLDOUT_SEED_START
from .time_penalty import TIME_PENALTY_COEFS

# The top of every schedule. 0.4 is the level the flat penalty collapses at,
# which is the point: a curriculum that arrives at 0.4 with a working policy
# is doing something a flat 0.4 cannot.
TARGET_PENALTY = 0.4


def linear_schedule(step: int, total_steps: int, target: float) -> float:
    """Ramp 0 -> target evenly across training, clamped past the budget.

    Clamping matters: SB3 finishes the rollout it's in, so `learn()` can run
    slightly past `total_timesteps` and an unclamped ramp would push the
    penalty above the level being tested.
    """
    return target * min(1.0, step / total_steps)


def stepped_schedule(step: int, total_steps: int, levels: Sequence[float]) -> float:
    """Hold each level for an equal slice of the budget, in order.

    Uses old roadmap Phase 4's own sweep levels by default, so every rung the
    curriculum passes through has an already-measured flat-penalty
    counterpart to compare against.
    """
    index = int(len(levels) * step / total_steps)
    return float(levels[min(index, len(levels) - 1)])


SCHEDULES: Dict[str, Callable[[int, float], Callable[[int], float]]] = {
    "linear": lambda total, target: (lambda t: linear_schedule(t, total, target)),
    "stepped": lambda total, target: (
        # Rescaled so the last rung is `target` rather than whatever the flat
        # sweep's top level happened to be -- every shape must share endpoints
        # or the comparison measures endpoints, not shape.
        lambda t: stepped_schedule(
            t, total, [lvl * target / TIME_PENALTY_COEFS[-1] for lvl in TIME_PENALTY_COEFS]
        )
    ),
}


class CurriculumTimePenaltyWrapper(gym.Wrapper):
    """Subtracts a scheduled, time-varying penalty from the reward each step.

    Unlike `TimePenaltyWrapper`, which is stateless per episode, this counts
    *total training steps across episodes* -- `reset()` deliberately does not
    zero the counter. Zeroing it would restart the schedule every episode and
    the penalty would never anneal at all, which is the single most likely way
    to get this wrapper subtly wrong.

    Used only when building the RL *training* env -- never for evaluation.
    """

    def __init__(self, env: gym.Env, schedule_fn: Callable[[int], float]):
        super().__init__(env)
        self.schedule_fn = schedule_fn
        self.total_steps = 0
        self.current_penalty = 0.0

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.total_steps += 1
        self.current_penalty = self.schedule_fn(self.total_steps)
        return observation, reward - self.current_penalty, terminated, truncated, info


def _train_and_eval_curriculum(args: tuple) -> Dict[str, float]:
    """Train one PPO model under a scheduled penalty, then score it on natural
    reward. Runs in a worker process."""
    shape, seed, total_timesteps, eval_episodes, target = args

    import torch

    # One thread per worker; torch's default of 4 oversubscribes the CPU
    # several times over once the pool is saturated (same as ppo_search).
    torch.set_num_threads(1)

    from ..controllers.rl_agent import RLAgent

    schedule_fn = SCHEDULES[shape](total_timesteps, target)
    agent = RLAgent()
    saved_path = agent.train(
        total_timesteps=total_timesteps,
        save_path=f"ppo_curriculum_{shape}_s{seed}",
        env_wrapper=lambda env: CurriculumTimePenaltyWrapper(env, schedule_fn),
        hyperparams={"seed": seed, "verbose": 0},
    )
    metrics = evaluate_controller_natural(agent, eval_episodes, seed_start=HOLDOUT_SEED_START)
    return {
        "shape": shape,
        "seed": seed,
        "target_penalty": target,
        "detail": saved_path,
        **metrics,
    }


def run_penalty_curriculum_sweep(
    shapes: Optional[List[str]] = None,
    seeds: Sequence[int] = (0,),
    total_timesteps: int = 1_000_000,
    eval_episodes: int = 100,
    target: float = TARGET_PENALTY,
    n_jobs: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Train one policy per (shape, seed) and score each on natural reward.

    Evaluation is always unpenalized, so the curriculum never scores its own
    effect -- same discipline as every other sweep here.
    """
    shapes = list(SCHEDULES) if shapes is None else shapes
    out_dir = Path(output_dir) if output_dir else new_run_dir("penalty_curriculum")
    out_dir.mkdir(parents=True, exist_ok=True)

    work = [(s, seed, total_timesteps, eval_episodes, target)
            for s in shapes for seed in seeds]
    n_jobs = min(n_jobs or os.cpu_count() or 1, len(work))
    print(f"Training {len(work)} PPO configs x {total_timesteps:,} timesteps "
          f"on {n_jobs} workers (target penalty {target})...")

    rows = []
    with multiprocessing.Pool(processes=n_jobs) as pool:
        for i, r in enumerate(pool.imap_unordered(_train_and_eval_curriculum, work), 1):
            rows.append(r)
            print(f"  [{i}/{len(work)}] {r['shape']} seed={r['seed']} "
                  f"reward={r['mean_reward']:.1f} success={r['success_rate_pct']:.0f}% "
                  f"steps={r['avg_landing_steps']:.1f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "penalty_curriculum_results.csv", index=False)
    print(f"\nResults: {out_dir / 'penalty_curriculum_results.csv'}")
    print(df.drop(columns=["detail"]).to_string(index=False))
    return df
