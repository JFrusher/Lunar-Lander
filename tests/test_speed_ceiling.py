"""Phase 1 of tmp/SPEED_ROADMAP.md: idealized point-mass minimum-time descent bound."""

import math

import pytest

from lunar_lander_lab.utils.speed_ceiling import (
    DescentConstants,
    measure_descent_constants,
    min_time_to_land,
    sample_initial_states,
    simulate_descent,
)


def test_simulate_descent_matches_closed_form_free_fall():
    """With switch_tick set unreachably late, the engine never fires and this
    is textbook free fall: t = sqrt(2h/g), touchdown speed = sqrt(2gh)."""
    const = DescentConstants(mass=5.0, gravity=-10.0, engine_dv_per_tick=1.5, dt=0.001)
    h0 = 20.0

    ticks, touchdown_speed = simulate_descent(
        switch_tick=10**9, h0=h0, v0=0.0, const=const, safe_touchdown_speed=999.0
    )

    expected_time = math.sqrt(2 * h0 / abs(const.gravity))
    expected_speed = math.sqrt(2 * abs(const.gravity) * h0)

    assert ticks * const.dt == pytest.approx(expected_time, rel=0.01)
    assert touchdown_speed == pytest.approx(expected_speed, rel=0.01)


def test_switch_tick_zero_lands_without_overshoot_even_for_a_strong_engine():
    """The real main engine measures stronger than gravity (net delta-v while
    firing is positive), so firing unconditionally can decelerate a fall past
    zero and into sustained ascent. Threshold-following avoids that -- it
    stops firing the instant speed is back under the cap -- so even the most
    conservative choice (threshold-following the whole flight) must still
    land, not fly away. Coarse ticks make the touchdown speed oscillate
    somewhat above the cap rather than hug it tightly, so this only checks
    that it's bounded, not tight."""
    const = DescentConstants(mass=5.0, gravity=-10.0, engine_dv_per_tick=1.5, dt=0.1)

    ticks, touchdown_speed = simulate_descent(
        switch_tick=0, h0=20.0, v0=0.0, const=const, safe_touchdown_speed=2.0, max_ticks=500
    )

    assert ticks < 500, "must land, not fly away"
    assert touchdown_speed < 10.0, "should be roughly near the cap, not still falling fast"


def test_max_ticks_bounds_the_loop():
    """If max_ticks is smaller than what's needed to land, the loop must stop
    there rather than run past it."""
    const = DescentConstants(mass=5.0, gravity=-10.0, engine_dv_per_tick=0.25, dt=0.02)

    ticks, _ = simulate_descent(
        switch_tick=0, h0=1_000_000.0, v0=0.0, const=const,
        safe_touchdown_speed=0.5, max_ticks=200,
    )

    assert ticks == 200


def test_min_time_to_land_is_within_cap_and_boundary_tight():
    """The chosen switch point should be the *latest* (most free-fall, hence
    fastest) one that still respects the cap -- one tick later must break it."""
    const = DescentConstants(mass=5.0, gravity=-10.0, engine_dv_per_tick=0.25, dt=0.02)

    result = min_time_to_land(h0=10.0, v0=0.0, const=const, safe_touchdown_speed=2.0)
    assert result["feasible"]
    assert result["touchdown_speed"] <= 2.0

    _, speed_one_tick_later = simulate_descent(
        switch_tick=result["switch_tick"] + 1, h0=10.0, v0=0.0, const=const,
        safe_touchdown_speed=2.0,
    )
    assert speed_one_tick_later > 2.0


def test_min_time_to_land_stricter_cap_takes_longer():
    const = DescentConstants(mass=5.0, gravity=-10.0, engine_dv_per_tick=0.25, dt=0.02)

    strict = min_time_to_land(h0=10.0, v0=0.0, const=const, safe_touchdown_speed=1.0)
    loose = min_time_to_land(h0=10.0, v0=0.0, const=const, safe_touchdown_speed=3.0)

    assert strict["feasible"] and loose["feasible"]
    assert strict["ticks"] > loose["ticks"]
    assert strict["touchdown_speed"] < loose["touchdown_speed"]


def test_min_time_to_land_reports_infeasible_when_no_switch_point_works():
    """A cap this tight can't be resolved at this dt's resolution -- the scan
    must exhaust cleanly and report failure rather than a bogus number."""
    const = DescentConstants(mass=5.0, gravity=-10.0, engine_dv_per_tick=0.25, dt=0.02)

    result = min_time_to_land(h0=10.0, v0=0.0, const=const, safe_touchdown_speed=0.01)

    assert not result["feasible"]


@pytest.mark.slow
def test_measure_descent_constants_returns_sane_values():
    const = measure_descent_constants(n_samples=5)
    assert const.mass > 0
    assert const.gravity < 0
    assert const.engine_dv_per_tick > 0, "main engine should net-push upward per tick"
    assert const.dt == pytest.approx(1 / 50)


@pytest.mark.slow
def test_sample_initial_states_returns_positive_altitudes():
    states = sample_initial_states(n_samples=5)

    assert len(states) == 5
    for h0, _v0 in states:
        assert h0 > 0, "lander should start above the touchdown line"
