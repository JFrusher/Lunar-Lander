"""Migrated from time_penalty.py's __main__ self-check, plus the wrapper's edges."""

import math

import gymnasium as gym
import pytest

from lunar_lander_lab.controllers.heuristic import HeuristicController
from lunar_lander_lab.utils.evaluation import VY_NORM_TO_WORLD, evaluate_controller_natural
from lunar_lander_lab.utils.time_penalty import EngineLockoutWrapper, TimePenaltyWrapper


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


def test_lockout_requires_exactly_one_gate_form():
    with pytest.raises(ValueError):
        EngineLockoutWrapper(gym.make("LunarLander-v3"))
    with pytest.raises(ValueError):
        EngineLockoutWrapper(
            gym.make("LunarLander-v3"), lockout_steps=5, altitude_threshold=0.5
        )


def test_step_gate_remaps_main_engine_to_noop_before_threshold():
    wrapped = EngineLockoutWrapper(gym.make("LunarLander-v3"), lockout_steps=3)
    plain = gym.make("LunarLander-v3")
    wrapped.reset(seed=0)
    plain.reset(seed=0)

    # tick 0 < lockout_steps=3: gated, action 2 must behave exactly like action 0
    obs_w, reward_w, *_ = wrapped.step(2)
    obs_p, reward_p, *_ = plain.step(0)
    wrapped.close()
    plain.close()

    assert reward_w == pytest.approx(reward_p)
    assert list(obs_w) == pytest.approx(list(obs_p))


def test_step_gate_lifts_at_threshold():
    wrapped = EngineLockoutWrapper(gym.make("LunarLander-v3"), lockout_steps=1)
    plain = gym.make("LunarLander-v3")
    wrapped.reset(seed=0)
    plain.reset(seed=0)

    wrapped.step(0)  # tick 0: gated, but action already 0 -- advances step count to 1
    plain.step(0)

    # tick 1 >= lockout_steps=1: gate lifted, action 2 fires for real
    obs_w, reward_w, *_ = wrapped.step(2)
    obs_p, reward_p, *_ = plain.step(2)
    wrapped.close()
    plain.close()

    assert reward_w == pytest.approx(reward_p)
    assert list(obs_w) == pytest.approx(list(obs_p))


def test_altitude_gate_remaps_main_engine_while_above_threshold():
    # A very low threshold: current altitude is always above it, so the gate
    # (active while altitude > threshold) never lifts during this episode.
    wrapped = EngineLockoutWrapper(gym.make("LunarLander-v3"), altitude_threshold=-999.0)
    plain = gym.make("LunarLander-v3")
    wrapped.reset(seed=0)
    plain.reset(seed=0)

    obs_w, reward_w, *_ = wrapped.step(2)
    obs_p, reward_p, *_ = plain.step(0)
    wrapped.close()
    plain.close()

    assert reward_w == pytest.approx(reward_p)
    assert list(obs_w) == pytest.approx(list(obs_p))


def test_altitude_gate_inactive_at_or_below_threshold():
    # A very high threshold: current altitude never exceeds it, so the gate
    # (active while altitude > threshold) is never active during this episode.
    wrapped = EngineLockoutWrapper(gym.make("LunarLander-v3"), altitude_threshold=999.0)
    plain = gym.make("LunarLander-v3")
    wrapped.reset(seed=0)
    plain.reset(seed=0)

    obs_w, reward_w, *_ = wrapped.step(2)
    obs_p, reward_p, *_ = plain.step(2)
    wrapped.close()
    plain.close()

    assert reward_w == pytest.approx(reward_p)
    assert list(obs_w) == pytest.approx(list(obs_p))


def test_inverted_altitude_gate_fires_below_the_threshold():
    """Arc 3 Phase D: a *terminal* cutoff, the mirror of Phase 2's initial
    lockout. Phase 8b found cushioning the touchdown doubles the settling
    tail, so cutting the engine for the last stretch tests whether a clean
    drop settles faster than a feathered one."""
    # Altitude is always above -999, so with invert the gate is never active.
    ungated = EngineLockoutWrapper(
        gym.make("LunarLander-v3"), altitude_threshold=-999.0, invert=True
    )
    plain = gym.make("LunarLander-v3")
    ungated.reset(seed=0)
    plain.reset(seed=0)

    _, reward_gated, *_ = ungated.step(2)
    _, reward_plain, *_ = plain.step(2)
    ungated.close()
    plain.close()

    assert reward_gated == pytest.approx(reward_plain)


def test_inverted_altitude_gate_blocks_when_below_threshold():
    # Altitude never exceeds 999, so with invert the gate is always active.
    gated = EngineLockoutWrapper(
        gym.make("LunarLander-v3"), altitude_threshold=999.0, invert=True
    )
    plain = gym.make("LunarLander-v3")
    gated.reset(seed=0)
    plain.reset(seed=0)

    obs_gated, reward_gated, *_ = gated.step(2)
    obs_plain, reward_plain, *_ = plain.step(0)
    gated.close()
    plain.close()

    assert reward_gated == pytest.approx(reward_plain)
    assert list(obs_gated) == pytest.approx(list(obs_plain))


def test_invert_is_rejected_for_a_step_gate():
    """Inverting a step gate would mean 'block after N steps', which is a
    different mechanism with no motivating hypothesis -- refuse rather than
    silently do something unintended."""
    with pytest.raises(ValueError, match="invert"):
        EngineLockoutWrapper(gym.make("LunarLander-v3"), lockout_steps=10, invert=True)


def test_evaluate_controller_natural_applies_env_wrapper():
    """The lockout sweep's heuristic leg wraps the *evaluation* env directly
    (no training involved) -- evaluate_controller_natural needs a hook for it,
    same shape as RLAgent.train's env_wrapper."""
    wrapped_metrics = evaluate_controller_natural(
        HeuristicController(),
        num_episodes=1,
        seed_start=0,
        env_wrapper=lambda env: EngineLockoutWrapper(env, altitude_threshold=-999.0),
    )
    plain_metrics = evaluate_controller_natural(
        HeuristicController(), num_episodes=1, seed_start=0
    )
    # A permanently-gated whole episode can't land the same way an ungated
    # one does -- just confirm the wrapper actually changed the outcome.
    assert wrapped_metrics != plain_metrics


class _FixedActionController:
    """Replays a known action sequence, then noops. Lets the fuel counters be
    checked against a count we control rather than whatever a real policy does."""

    def __init__(self, actions):
        self.actions = list(actions)
        self.index = 0

    def get_action(self, observation):
        action = self.actions[self.index] if self.index < len(self.actions) else 0
        self.index += 1
        return action


def test_fuel_counters_match_the_actions_actually_taken():
    """The env charges 0.3/frame for the main engine and 0.03/frame for a side
    engine, so fuel is already inside every reward number in this repo --
    silently. These counters make it visible."""
    actions = [2, 2, 2, 1, 3, 0]  # 3 main, 2 side, 1 noop
    metrics = evaluate_controller_natural(
        _FixedActionController(actions), num_episodes=1, seed_start=0
    )

    assert metrics["main_engine_frames"] == 3
    assert metrics["side_engine_frames"] == 2
    assert metrics["fuel_cost"] == pytest.approx(3 * 0.3 + 2 * 0.03)


def test_touchdown_speed_is_reported_for_landings():
    """Phase 1's ceiling model is parameterised entirely by touchdown speed
    but no real episode has ever reported it. Positive-down, matching
    speed_ceiling.simulate_descent's convention."""
    metrics = evaluate_controller_natural(HeuristicController(), num_episodes=3, seed_start=0)

    assert "avg_touchdown_speed" in metrics
    assert not math.isnan(metrics["avg_touchdown_speed"]), "should land at least once in 3"
    assert metrics["avg_touchdown_speed"] > 0, "descending at touchdown is positive-down"


def test_touchdown_speed_is_in_world_units_not_normalized():
    """observation[3] is normalized by the env; speed_ceiling works in world
    units. Reporting the raw value would make the lander look 7.5x gentler
    than it lands and silently break the comparison the metric exists for."""
    assert VY_NORM_TO_WORLD == pytest.approx(7.5)

    metrics = evaluate_controller_natural(HeuristicController(), num_episodes=5, seed_start=10_000)

    # Normalized touchdown is ~0.18; world units put it near 1.4. Anything
    # below 1.0 means the conversion was dropped somewhere.
    assert metrics["avg_touchdown_speed"] > 1.0


def test_landing_steps_split_into_flight_and_settling():
    """An episode does not end at touchdown -- it ends once Box2D puts the
    lander to sleep. Everything after first leg contact is time the
    controller can no longer influence by flying better, so reporting only
    the total hides an incompressible chunk of it."""
    metrics = evaluate_controller_natural(HeuristicController(), num_episodes=5, seed_start=50_000)

    assert metrics["avg_flight_steps"] + metrics["avg_settle_steps"] == pytest.approx(
        metrics["avg_landing_steps"]
    )
    assert metrics["avg_settle_steps"] > 0, "there is always some post-contact settling"
    assert metrics["avg_flight_steps"] > metrics["avg_settle_steps"]


def test_new_metrics_are_additive_and_break_no_existing_key():
    """Every prior result in this repo was computed with these four keys.
    Adding columns must not rename or change any of them."""
    metrics = evaluate_controller_natural(HeuristicController(), num_episodes=2, seed_start=0)

    assert {
        "mean_reward",
        "success_rate_pct",
        "crash_rate_pct",
        "avg_landing_steps",
    } <= set(metrics)


@pytest.mark.slow
def test_evaluate_controller_natural_reports_all_metrics():
    metrics = evaluate_controller_natural(HeuristicController(), num_episodes=3)
    assert set(metrics) == {
        "mean_reward",
        "success_rate_pct",
        "crash_rate_pct",
        "avg_landing_steps",
        "avg_touchdown_speed",
        "avg_flight_steps",
        "avg_settle_steps",
        "main_engine_frames",
        "side_engine_frames",
        "fuel_cost",
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
