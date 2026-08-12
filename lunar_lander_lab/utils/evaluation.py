"""Evaluation: the shared definition of how a controller is scored.

Holds the success criterion, the engine/fuel accounting, and
`evaluate_controller_natural` -- the function nearly every study in this repo
uses to turn a controller into numbers. These lived in `time_penalty.py`
until they outgrew it: that module is named after one reward-shaping
experiment, and it had also grown a second, duplicate definition of
`SUCCESS_REWARD_THRESHOLD`, so the success criterion had two sources of
truth.
"""

import math
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium.envs.box2d.lunar_lander import FPS, SCALE, VIEWPORT_H

from .paths import new_run_dir

if TYPE_CHECKING:  # pragma: no cover
    # Type-checking only. Importing the controllers package at runtime creates
    # a cycle -- controllers/__init__ pulls in ScheduledHeuristicController,
    # which builds its parameter space from pid_search, which imports this
    # module. Worse, the cycle only fires on some import orders, and in a
    # multiprocessing pool the workers die on import and the parent waits
    # forever rather than reporting anything.
    from ..controllers.base import BaseController

# A landing is considered successful once the episode's total reward crosses
# this threshold (LunarLander awards +100 for a safe landing on top of the
# shaping reward accrued during descent).
SUCCESS_REWARD_THRESHOLD = 200
# A crash is signalled by the large one-off penalty LunarLander applies when
# the lander body collides with the ground/moon.
CRASH_REWARD_THRESHOLD = 100

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


def evaluate_controller_natural(
    controller: "BaseController",
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


def run_benchmark(
    controllers: Dict[str, "BaseController"],
    env_name: str = "LunarLander-v3",
    num_episodes: int = 50,
    plot_path: Optional[str] = None,
) -> pd.DataFrame:
    """Evaluate each controller over the same set of episode seeds.

    Returns a DataFrame of per-controller summary metrics and writes a
    comparison bar chart to ``plot_path`` (default: a new
    runs/benchmark/<timestamp>/benchmark_results.png).
    """
    if plot_path is None:
        plot_path = str(new_run_dir("benchmark") / "benchmark_results.png")

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
