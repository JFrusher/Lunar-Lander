"""Migrated from time_penalty.py's __main__ self-check, plus the wrapper's edges."""

import math

import gymnasium as gym
import pytest

from lunar_lander_lab.controllers.heuristic import HeuristicController
from lunar_lander_lab.utils.time_penalty import (
    TimePenaltyWrapper,
    evaluate_controller_natural,
)


def test_wrapper_subtracts_exactly_the_penalty_each_step():
    plain = gym.make("LunarLander-v3")
    penalized = TimePenaltyWrapper(gym.make("LunarLander-v3"), penalty_per_step=0.05)
    plain.reset(seed=0)
    penalized.reset(seed=0)

    _, reward_plain, *_ = plain.step(0)
    _, reward_penalized, *_ = penalized.step(0)
    plain.close()
    penalized.close()

    assert reward_plain - 0.05 == pytest.approx(reward_penalized)


def test_zero_penalty_leaves_reward_untouched():
    plain = gym.make("LunarLander-v3")
    penalized = TimePenaltyWrapper(gym.make("LunarLander-v3"), penalty_per_step=0.0)
    plain.reset(seed=0)
    penalized.reset(seed=0)

    _, reward_plain, *_ = plain.step(2)
    _, reward_penalized, *_ = penalized.step(2)
    plain.close()
    penalized.close()

    assert reward_plain == pytest.approx(reward_penalized)


@pytest.mark.slow
def test_evaluate_controller_natural_reports_all_metrics():
    metrics = evaluate_controller_natural(HeuristicController(), num_episodes=3)
    assert set(metrics) == {
        "mean_reward",
        "success_rate_pct",
        "crash_rate_pct",
        "avg_landing_steps",
    }
    assert not math.isnan(metrics["avg_landing_steps"]), "should land at least once in 3 tries"


@pytest.mark.slow
def test_evaluate_controller_natural_is_seed_reproducible():
    a = evaluate_controller_natural(HeuristicController(), num_episodes=3)
    b = evaluate_controller_natural(HeuristicController(), num_episodes=3)
    assert a["mean_reward"] == b["mean_reward"]


@pytest.mark.slow
def test_seed_start_selects_a_different_episode_set():
    a = evaluate_controller_natural(HeuristicController(), num_episodes=3, seed_start=0)
    b = evaluate_controller_natural(HeuristicController(), num_episodes=3, seed_start=500)
    assert a["mean_reward"] != b["mean_reward"]
