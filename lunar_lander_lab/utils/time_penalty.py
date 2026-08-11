"""Time-penalty reward shaping + sweep across controller types."""

import json
import math
import multiprocessing
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium.envs.box2d.lunar_lander import FPS, SCALE, VIEWPORT_H

from ..controllers.base import BaseController
from ..controllers.rl_agent import RLAgent
from .paths import new_run_dir
from .pid_search import HOLDOUT_SEED_START, run_monte_carlo

SUCCESS_REWARD_THRESHOLD = 200

TIME_PENALTY_COEFS: List[float] = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]

MAIN_ENGINE_ACTION = 2
SIDE_ENGINE_ACTIONS = (1, 3)

# Phase 8b: the same env with a Box(-1, 1, (2,)) action space instead of
# Discrete(4). Observation space is identical, which is what makes the
# comparison a single-variable one.
CONTINUOUS_ENV_KWARGS = {"continuous": True}

# The env's own firing thresholds for continuous actions (see LunarLander's
# step()): the main engine is off for action[0] <= 0, and a side booster only
# fires for |action[1]| > 0.5. Counting frames on any other threshold would
# bill fuel that never burned.
CONTINUOUS_MAIN_THRESHOLD = 0.0
CONTINUOUS_SIDE_THRESHOLD = 0.5


def count_engine_frames(action) -> Tuple[int, int]:
    """Return (main_engine_frames, side_engine_frames) for one action.

    Handles both action spaces so fuel accounting is comparable across the
    discrete/continuous split rather than silently zero on one side.
    """
    if np.isscalar(action) or getattr(action, "ndim", 0) == 0:
        value = int(action)
        return int(value == MAIN_ENGINE_ACTION), int(value in SIDE_ENGINE_ACTIONS)
    return (
        int(action[0] > CONTINUOUS_MAIN_THRESHOLD),
        int(abs(action[1]) > CONTINUOUS_SIDE_THRESHOLD),
    )

# observation[3] is vertical velocity *normalized* by the env
# (state[3] = vel.y * (VIEWPORT_H/SCALE/2) / FPS), not world units. Touchdown
# speed is reported in world units so it is directly comparable with
# `speed_ceiling`, which solves the descent in world units -- mixing the two
# scales silently makes a controller look 7.5x gentler than it lands.
_VY_NORMALIZER = (VIEWPORT_H / SCALE / 2) / FPS
VY_NORM_TO_WORLD = 1.0 / _VY_NORMALIZER
# LunarLander's own per-frame reward charges for firing (see its docstring:
# "decreased by 0.3 points each frame the main engine is firing", 0.03 for a
# side engine). Reusing the env's own prices keeps `fuel_cost` on the same
# scale as the reward it's already silently deducted from.
MAIN_ENGINE_REWARD_COST = 0.3
SIDE_ENGINE_REWARD_COST = 0.03

# Fixed-order categorical pair (dataviz skill default palette, slots 1-2:
# blue/orange), validated CVD-safe adjacent pair in both light and dark modes.
_CONTROLLER_COLORS = {"Heuristic": "#2a78d6", "RL (PPO)": "#eb6834"}
_SURFACE = "#fcfcfb"
_GRID = "#e1e0d9"
_INK_PRIMARY = "#0b0b0b"
_INK_MUTED = "#898781"


class TimePenaltyWrapper(gym.Wrapper):
    """Subtracts a fixed penalty from the reward every timestep.

    Used only when building the RL *training* env — never for evaluation.
    """

    def __init__(self, env: gym.Env, penalty_per_step: float):
        super().__init__(env)
        self.penalty_per_step = penalty_per_step

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward - self.penalty_per_step, terminated, truncated, info


class EngineLockoutWrapper(gym.Wrapper):
    """Remaps the main engine (action 2) to noop (0) while a gate holds.

    Exactly one of `lockout_steps` or `altitude_threshold` must be set --
    these are two alternative gate forms being compared against each other
    (tmp/SPEED_ROADMAP.md Phase 2), not combined into one wrapper.
    """

    def __init__(
        self,
        env: gym.Env,
        lockout_steps: Optional[int] = None,
        altitude_threshold: Optional[float] = None,
    ):
        super().__init__(env)
        if (lockout_steps is None) == (altitude_threshold is None):
            raise ValueError("set exactly one of lockout_steps or altitude_threshold")
        self.lockout_steps = lockout_steps
        self.altitude_threshold = altitude_threshold
        self._step_count = 0
        self._altitude = 0.0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._step_count = 0
        self._altitude = observation[1]
        return observation, info

    def step(self, action):
        if action == 2 and self._gate_active():
            action = 0
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        self._altitude = observation[1]
        return observation, reward, terminated, truncated, info

    def _gate_active(self) -> bool:
        if self.lockout_steps is not None:
            return self._step_count < self.lockout_steps
        return self._altitude > self.altitude_threshold


def evaluate_controller_natural(
    controller: BaseController,
    num_episodes: int,
    env_name: str = "LunarLander-v3",
    seed_start: int = 0,
    env_wrapper: Optional[Callable[[gym.Env], gym.Env]] = None,
    env_kwargs: Optional[Dict] = None,
) -> Dict[str, float]:
    """Evaluate `controller` over `num_episodes` seeded episodes with natural (unpenalized) reward.

    Also reports touchdown speed and fuel burn. Both were previously
    invisible: touchdown speed is the quantity `speed_ceiling`'s whole model
    is parameterised by yet no real episode ever measured, and fuel is
    already priced into every reward number in this repo (the env charges per
    engine-frame) without ever being broken out. A controller that gets
    faster by burning more fuel is a different result from one that flies a
    better trajectory, and without these the two are indistinguishable.
    """
    env = gym.make(env_name, **(env_kwargs or {}))
    if env_wrapper:
        env = env_wrapper(env)
    successes = crashes = 0
    main_frames = side_frames = 0
    rewards = []
    success_steps = []
    touchdown_speeds = []
    flight_steps = []
    settle_steps = []

    for seed in range(seed_start, seed_start + num_episodes):
        observation, _ = env.reset(seed=seed)
        total_reward = 0.0
        steps = 0
        terminated = truncated = False

        impact_speed = math.nan
        first_contact_step = None
        while not (terminated or truncated):
            action = controller.get_action(observation)
            main, side = count_engine_frames(action)
            main_frames += main
            side_frames += side
            # Vertical velocity going *into* this step, positive-up. Held
            # because the episode does not end at touchdown -- it ends once
            # the lander has settled and Box2D puts it to sleep, by which
            # point the final observation reads exactly 0 velocity. Impact
            # speed has to be sampled at first leg contact or not at all.
            approach_vy = float(observation[3])
            had_contact = bool(observation[6] or observation[7])

            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            steps += 1

            if math.isnan(impact_speed) and not had_contact and (observation[6] or observation[7]):
                # Negated for speed_ceiling.simulate_descent's positive-down
                # convention, and de-normalized into world units so the two
                # are on the same scale.
                impact_speed = -approach_vy * VY_NORM_TO_WORLD
                first_contact_step = steps

        rewards.append(total_reward)
        if total_reward >= SUCCESS_REWARD_THRESHOLD:
            successes += 1
            success_steps.append(steps)
            if not math.isnan(impact_speed):
                touchdown_speeds.append(impact_speed)
            if first_contact_step is not None:
                flight_steps.append(first_contact_step)
                settle_steps.append(steps - first_contact_step)
        elif terminated and total_reward <= -100:
            crashes += 1

    env.close()

    return {
        "mean_reward": sum(rewards) / len(rewards),
        "success_rate_pct": 100 * successes / num_episodes,
        "crash_rate_pct": 100 * crashes / num_episodes,
        "avg_landing_steps": (
            sum(success_steps) / len(success_steps) if success_steps else math.nan
        ),
        "avg_touchdown_speed": (
            sum(touchdown_speeds) / len(touchdown_speeds) if touchdown_speeds else math.nan
        ),
        # avg_landing_steps split at first leg contact. Everything after it is
        # Box2D settling the lander to sleep -- real episode time, charged by
        # any time penalty, but not something the controller can fly its way
        # out of. Reporting only the total conflates a compressible quantity
        # with an incompressible one.
        "avg_flight_steps": (
            sum(flight_steps) / len(flight_steps) if flight_steps else math.nan
        ),
        "avg_settle_steps": (
            sum(settle_steps) / len(settle_steps) if settle_steps else math.nan
        ),
        "main_engine_frames": main_frames,
        "side_engine_frames": side_frames,
        "fuel_cost": main_frames * MAIN_ENGINE_REWARD_COST
        + side_frames * SIDE_ENGINE_REWARD_COST,
    }


def _style_axes(*axes) -> None:
    for ax in axes:
        ax.set_facecolor(_SURFACE)
        ax.grid(True, color=_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color(_GRID)
        ax.tick_params(colors=_INK_MUTED)


def _plot_tradeoff(df: pd.DataFrame, path: Path) -> None:
    fig, (ax_steps, ax_reward) = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=_SURFACE)
    _style_axes(ax_steps, ax_reward)

    for controller, group in df.groupby("controller"):
        group = group.sort_values("penalty")
        color = _CONTROLLER_COLORS.get(controller, "#4a3aa7")
        ax_steps.plot(
            group["penalty"], group["avg_landing_steps"],
            marker="o", markersize=8, linewidth=2, color=color, label=controller,
        )
        ax_reward.plot(
            group["penalty"], group["mean_reward"],
            marker="o", markersize=8, linewidth=2, color=color, label=controller,
        )

    ax_steps.set_xlabel("time penalty (per step)", color=_INK_PRIMARY)
    ax_steps.set_ylabel("avg landing steps (successful episodes)", color=_INK_PRIMARY)
    ax_steps.set_title("Landing Speed vs. Penalty", color=_INK_PRIMARY)

    ax_reward.set_xlabel("time penalty (per step)", color=_INK_PRIMARY)
    ax_reward.set_ylabel("mean reward (natural)", color=_INK_PRIMARY)
    ax_reward.set_title("Reward vs. Penalty", color=_INK_PRIMARY)

    handles, labels = ax_steps.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06), frameon=False)

    fig.suptitle("Time-Penalty Sweep: Landing Speed / Reward Trade-off", color=_INK_PRIMARY, y=1.16)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=_SURFACE, bbox_inches="tight")
    plt.close(fig)


def run_time_penalty_sweep(
    penalties: List[float] = TIME_PENALTY_COEFS,
    rl_timesteps: int = 150_000,
    pid_samples: int = 200,
    pid_episodes: int = 30,
    eval_episodes: int = 30,
    seed: int = 0,
    n_jobs: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Sweep `penalties`, training/tuning both controllers at each level, and
    report natural-reward metrics + a trade-off plot."""
    sweep_dir = Path(output_dir) if output_dir else new_run_dir("time_penalty_sweep")
    rows = []

    for i, p in enumerate(penalties):
        print(f"\n=== Heuristic, penalty={p} ===")
        run_dir = sweep_dir / f"heuristic_penalty_{p}"
        run_monte_carlo(
            n_samples=pid_samples,
            episodes_per_set=pid_episodes,
            seed=seed + i,
            time_penalty=p,
            output_dir=str(run_dir),
            n_jobs=n_jobs,
            holdout_episodes=eval_episodes,
        )
        # Read the winner run_monte_carlo already picked on held-out episodes.
        # Re-deriving it here from the returned frame would re-introduce the
        # search-set-only argmax that selection is meant to avoid.
        best = json.loads((run_dir / "best_gains.json").read_text())
        rows.append(
            {
                "controller": "Heuristic",
                "penalty": p,
                "mean_reward": best["mean_reward"],
                "success_rate_pct": best["success_rate_pct"],
                "crash_rate_pct": best["crash_rate_pct"],
                "avg_landing_steps": best["avg_steps_success"],
                "detail": str(run_dir / "best_gains.json"),
            }
        )

    for p in penalties:
        print(f"\n=== RL (PPO), penalty={p} ===")
        agent = RLAgent()
        saved_path = agent.train(
            total_timesteps=rl_timesteps,
            save_path=f"ppo_penalty_{p}",
            env_wrapper=lambda env, p=p: TimePenaltyWrapper(env, p),
            # Same PPO seed at every penalty level (not varied like the heuristic's
            # search seed above) — isolates the penalty as the only thing that
            # differs between runs, instead of confounding it with random network
            # init/rollout noise.
            hyperparams={"seed": seed},
        )
        # Same episodes the heuristic leg is now scored on, so the two lines on
        # the trade-off plot are measured against an identical set of landings.
        metrics = evaluate_controller_natural(
            agent, eval_episodes, seed_start=HOLDOUT_SEED_START
        )
        rows.append({"controller": "RL (PPO)", "penalty": p, "detail": saved_path, **metrics})

    df_results = pd.DataFrame(rows)
    csv_path = sweep_dir / "time_penalty_sweep_results.csv"
    df_results.to_csv(csv_path, index=False)

    plot_path = sweep_dir / "time_penalty_tradeoff.png"
    _plot_tradeoff(df_results, plot_path)

    print(f"\nResults: {csv_path}")
    print(f"Plot:    {plot_path}")
    print(df_results.to_string(index=False))
    return df_results


def aggregate_seeds(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-seed rows into mean ± std per (controller, penalty).

    Flattens pandas' MultiIndex columns to `<metric>_mean` / `<metric>_std`,
    which is the contract `_plot_tradeoff_with_error_bars` reads.
    """
    metrics = ["mean_reward", "success_rate_pct", "crash_rate_pct", "avg_landing_steps"]
    agg = df_raw.groupby(["controller", "penalty"])[metrics].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(c).rstrip("_") for c in agg.columns]
    return agg


def _train_and_eval_rl(args: tuple) -> Dict[str, float]:
    """Train one PPO model at (seed, penalty) and evaluate it. Runs in a worker."""
    seed, penalty, rl_timesteps, eval_episodes = args

    import torch

    # One thread per worker; torch's default of 4 oversubscribes the CPU
    # several times over once the pool is saturated (same as ppo_search).
    torch.set_num_threads(1)

    agent = RLAgent()
    saved_path = agent.train(
        total_timesteps=rl_timesteps,
        save_path=f"ppo_p{penalty}_s{seed}",
        env_wrapper=lambda env, p=penalty: TimePenaltyWrapper(env, p),
        hyperparams={"seed": seed, "verbose": 0},
    )
    metrics = evaluate_controller_natural(agent, eval_episodes, seed_start=HOLDOUT_SEED_START)
    return {
        "controller": "RL (PPO)",
        "penalty": penalty,
        "seed": seed,
        "detail": saved_path,
        **metrics,
    }


def _plot_tradeoff_with_error_bars(agg: pd.DataFrame, path: Path, n_seeds: int) -> None:
    fig, (ax_steps, ax_reward) = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=_SURFACE)
    _style_axes(ax_steps, ax_reward)

    for controller, group in agg.groupby("controller"):
        group = group.sort_values("penalty")
        color = _CONTROLLER_COLORS.get(controller, "#4a3aa7")
        for ax, metric in ((ax_steps, "avg_landing_steps"), (ax_reward, "mean_reward")):
            ax.errorbar(
                group["penalty"], group[f"{metric}_mean"], yerr=group[f"{metric}_std"],
                marker="o", markersize=7, linewidth=2, color=color, label=controller,
                capsize=4, elinewidth=1.4,
            )

    ax_steps.set_xlabel("time penalty (per step)", color=_INK_PRIMARY)
    ax_steps.set_ylabel("avg landing steps (successful episodes)", color=_INK_PRIMARY)
    ax_steps.set_title("Landing Speed vs. Penalty", color=_INK_PRIMARY)

    ax_reward.set_xlabel("time penalty (per step)", color=_INK_PRIMARY)
    ax_reward.set_ylabel("mean reward (natural, held-out)", color=_INK_PRIMARY)
    ax_reward.set_title("Reward vs. Penalty", color=_INK_PRIMARY)

    handles, labels = ax_steps.get_legend_handles_labels()
    fig.legend(handles[:2], labels[:2], loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.06), frameon=False)
    fig.suptitle(
        f"Time-Penalty Sweep: Landing Speed / Reward Trade-off "
        f"(mean ± std over {n_seeds} seeds)",
        color=_INK_PRIMARY, y=1.16,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=_SURFACE, bbox_inches="tight")
    plt.close(fig)


def run_multi_seed_sweep(
    penalties: List[float] = TIME_PENALTY_COEFS,
    seeds: Sequence[int] = (0, 100, 200),
    rl_timesteps: int = 1_000_000,
    pid_samples: int = 200,
    pid_episodes: int = 30,
    eval_episodes: int = 100,
    n_jobs: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Repeat the time-penalty sweep across `seeds` and aggregate mean ± std.

    Single point-estimates have already misled twice in this project (the
    fixed-seed Latin Hypercube, and PPO trained below its convergence floor),
    so the headline trade-off chart gets error bars rather than one line.

    Deviates from the roadmap's "just call run_time_penalty_sweep N times":
    that trains PPO serially, which measured out at ~6 h for 18 models. The
    RL leg is pooled across (seed, penalty) instead — ~2 h for the same work.
    """
    sweep_dir = Path(output_dir) if output_dir else new_run_dir("multi_seed_sweep")
    n_jobs = n_jobs or os.cpu_count() or 1
    rows: List[Dict[str, float]] = []

    for base_seed in seeds:
        for i, p in enumerate(penalties):
            print(f"\n=== Heuristic, penalty={p}, seed={base_seed} ===")
            run_dir = sweep_dir / f"heuristic_p{p}_s{base_seed}"
            run_monte_carlo(
                n_samples=pid_samples,
                episodes_per_set=pid_episodes,
                # Search seed varies per penalty level AND per repeat — the
                # Phase 0 fix, extended so repeats aren't identical either.
                seed=base_seed + i,
                time_penalty=p,
                output_dir=str(run_dir),
                n_jobs=n_jobs,
                holdout_episodes=eval_episodes,
            )
            best = json.loads((run_dir / "best_gains.json").read_text())
            rows.append(
                {
                    "controller": "Heuristic",
                    "penalty": p,
                    "seed": base_seed,
                    "mean_reward": best["mean_reward"],
                    "success_rate_pct": best["success_rate_pct"],
                    "crash_rate_pct": best["crash_rate_pct"],
                    "avg_landing_steps": best["avg_steps_success"],
                    "detail": str(run_dir / "best_gains.json"),
                }
            )

    rl_items = [
        (base_seed, p, rl_timesteps, eval_episodes) for base_seed in seeds for p in penalties
    ]
    print(f"\n=== RL (PPO): {len(rl_items)} models x {rl_timesteps:,} timesteps "
          f"on {n_jobs} workers ===")
    with multiprocessing.Pool(processes=n_jobs) as pool:
        for i, result in enumerate(pool.imap_unordered(_train_and_eval_rl, rl_items), 1):
            rows.append(result)
            print(f"  [{i}/{len(rl_items)}] penalty={result['penalty']} "
                  f"seed={result['seed']} reward={result['mean_reward']:.1f}")

    df_raw = pd.DataFrame(rows)
    df_raw.to_csv(sweep_dir / "multi_seed_raw.csv", index=False)

    agg = aggregate_seeds(df_raw)
    agg.to_csv(sweep_dir / "multi_seed_aggregated.csv", index=False)

    plot_path = sweep_dir / "time_penalty_tradeoff_errorbars.png"
    _plot_tradeoff_with_error_bars(agg, plot_path, n_seeds=len(seeds))

    print(f"\nRaw:        {sweep_dir / 'multi_seed_raw.csv'}")
    print(f"Aggregated: {sweep_dir / 'multi_seed_aggregated.csv'}")
    print(f"Plot:       {plot_path}")
    print(agg.to_string(index=False))
    return agg
