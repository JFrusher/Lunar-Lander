"""Arc 3 Phase A: does a 2-D bound explain the 1.5x gap the 1-D bound left?"""

import math

import pytest

from lunar_lander_lab.utils.speed_ceiling import (
    DescentConstants,
    PlanarConstants,
    measure_planar_constants,
    min_time_lateral,
    planar_bound,
    sample_planar_initial_states,
)

# Deliberately round numbers so the expected values are hand-checkable.
CONST = PlanarConstants(
    descent=DescentConstants(mass=5.0, gravity=-10.0, engine_dv_per_tick=0.36, dt=0.02),
    side_dv_per_tick=0.05,
    side_domega_per_tick=0.095,
    inertia=0.83,
)


def test_lateral_time_matches_closed_form_double_integrator():
    """Bringing (x, v) -> (0, 0) with |a| <= A from rest at distance d is the
    textbook accelerate-then-decelerate profile: t = 2*sqrt(d/A)."""
    accel_per_tick = 0.04
    a = accel_per_tick / CONST.descent.dt  # per second^2
    d = 2.0

    ticks = min_time_lateral(x0=d, vx0=0.0, accel_per_tick=accel_per_tick, dt=CONST.descent.dt)

    assert ticks * CONST.descent.dt == pytest.approx(2 * math.sqrt(d / a), rel=0.05)


def test_lateral_time_is_zero_when_already_at_rest_on_target():
    assert min_time_lateral(x0=0.0, vx0=0.0, accel_per_tick=0.04, dt=0.02) == 0


def test_lateral_time_grows_with_initial_velocity():
    """The measured start states carry up to ~4 u/s of lateral velocity, which
    is the part of the horizontal problem that actually costs time -- the
    lander starts nearly centred."""
    slow = min_time_lateral(x0=0.0, vx0=1.0, accel_per_tick=0.04, dt=0.02)
    fast = min_time_lateral(x0=0.0, vx0=4.0, accel_per_tick=0.04, dt=0.02)

    assert fast > slow > 0


def test_stronger_thrust_is_never_slower():
    weak = min_time_lateral(x0=1.0, vx0=2.0, accel_per_tick=0.02, dt=0.02)
    strong = min_time_lateral(x0=1.0, vx0=2.0, accel_per_tick=0.08, dt=0.02)

    assert strong <= weak


def test_untilted_strategy_is_at_least_the_vertical_leg():
    """Without tilt the lander must finish both the descent and the lateral
    correction, so the strategy takes the longer of the two legs."""
    result = planar_bound(
        h0=9.4, vy0=-1.0, x0=0.05, vx0=3.0, const=CONST, safe_touchdown_speed=1.42
    )

    assert result["untilted_ticks"] >= result["vertical_ticks"]
    assert result["untilted_ticks"] >= result["lateral_ticks"]


def test_best_ticks_is_the_better_of_the_two_strategies():
    """These are achievable strategies, not lower bounds, so the reported
    figure is whichever plan actually lands soonest."""
    result = planar_bound(
        h0=9.4, vy0=-1.0, x0=0.05, vx0=3.0, const=CONST, safe_touchdown_speed=1.42
    )

    assert result["best_ticks"] == min(result["untilted_ticks"], result["best_tilt_ticks"])


def test_centred_and_still_reduces_to_the_vertical_problem():
    """No lateral error means no lateral work, so the untilted strategy
    collapses to the 1-D answer -- the degenerate case proving this is a
    strict generalisation of the original model."""
    result = planar_bound(
        h0=9.4, vy0=-1.0, x0=0.0, vx0=0.0, const=CONST, safe_touchdown_speed=1.42
    )

    assert result["lateral_ticks"] == 0
    assert result["untilted_ticks"] == result["vertical_ticks"]


def test_tilt_search_rejects_infeasible_angles():
    """Past some tilt the vertical component drops below gravity and the
    lander cannot land at all. Such angles must be skipped, not scored -- an
    unchecked search picks them precisely because falling out of the sky
    reaches the ground quickly."""
    result = planar_bound(
        h0=9.4, vy0=-1.0, x0=0.05, vx0=3.5, const=CONST, safe_touchdown_speed=1.42
    )

    assert 0.0 <= result["tilt_rad"] < math.pi / 2
    # A landing that fast would only be reachable by crashing.
    assert result["best_tilt_ticks"] > 0


@pytest.mark.slow
def test_measure_planar_constants_matches_the_live_env():
    const = measure_planar_constants(n_samples=5)

    assert const.descent.engine_dv_per_tick > 0
    assert const.side_dv_per_tick > 0, "side engine should produce lateral thrust"
    assert const.side_domega_per_tick > 0, "side engine should produce torque"
    assert const.inertia > 0


@pytest.mark.slow
def test_sampled_start_states_carry_real_lateral_velocity():
    """The random initial push is the reason a 2-D bound differs from a 1-D
    one at all -- if the lander started still and centred there'd be nothing
    to correct."""
    states = sample_planar_initial_states(n_samples=10)

    assert len(states) == 10
    assert max(abs(vx0) for _, _, _, vx0 in states) > 1.0
