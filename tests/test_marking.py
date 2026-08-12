"""Arc 4: a modular marking framework.

Every major finding in this project came from decomposing a scalar -- flight
vs settling, gains per altitude band, success per physics condition. This
generalises that: segment an episode, grade each part against what the
idealized plan achieves in the same part, and detect named behaviours.
"""

import numpy as np
import pandas as pd
import pytest

from lunar_lander_lab.utils.marking import (
    SEGMENTS,
    behaviour_scores,
    grade,
    segment_of,
    summarise_segments,
)


def _steps(rows):
    """Build a per-step trace frame. Columns match record_episode's output."""
    return pd.DataFrame(
        rows,
        columns=["y", "vy", "x", "vx", "angle", "omega", "contact", "action"],
    )


def test_segments_are_ordered_high_to_low_and_end_with_settling():
    assert SEGMENTS[-1] == "SETTLING"
    assert SEGMENTS[0] == "DESCENT"


def test_segment_of_uses_altitude_while_airborne():
    assert segment_of(y=1.0, contact=False) == "DESCENT"
    assert segment_of(y=0.3, contact=False) == "APPROACH"
    assert segment_of(y=0.05, contact=False) == "TERMINAL"


def test_contact_overrides_altitude():
    """Once a leg is down the lander is settling regardless of the altitude
    reading -- this is the flight/settling split that produced Arc 2's
    incompressibility finding, and it is event-based, not altitude-based."""
    assert segment_of(y=1.0, contact=True) == "SETTLING"
    assert segment_of(y=0.0, contact=True) == "SETTLING"


def test_summarise_counts_ticks_and_fuel_per_segment():
    steps = _steps([
        [1.0, -0.5, 0, 0, 0, 0, False, 2],   # DESCENT, main engine
        [0.9, -0.5, 0, 0, 0, 0, False, 0],   # DESCENT, coast
        [0.3, -0.5, 0, 0, 0, 0, False, 1],   # APPROACH, side engine
        [0.0, -0.1, 0, 0, 0, 0, True, 0],    # SETTLING
    ])

    out = summarise_segments(steps)

    assert out["DESCENT"]["ticks"] == 2
    assert out["APPROACH"]["ticks"] == 1
    assert out["SETTLING"]["ticks"] == 1
    assert out["TERMINAL"]["ticks"] == 0
    assert out["DESCENT"]["fuel"] == pytest.approx(0.3)
    assert out["APPROACH"]["fuel"] == pytest.approx(0.03)


def test_hovering_counts_slow_descent_while_still_high():
    """The original complaint that started this whole project, finally
    measured as a behaviour rather than inferred from total step count."""
    hovering = _steps([[1.0, -0.01, 0, 0, 0, 0, False, 2]] * 10)
    falling = _steps([[1.0, -2.0, 0, 0, 0, 0, False, 0]] * 10)

    assert behaviour_scores(hovering)["hover_ticks"] == 10
    assert behaviour_scores(falling)["hover_ticks"] == 0


def test_hovering_ignores_slow_movement_near_the_ground():
    """Descending slowly just above the pad is the correct terminal
    behaviour, not dawdling. Counting it would penalise a good landing."""
    near_ground = _steps([[0.02, -0.01, 0, 0, 0, 0, False, 2]] * 10)

    assert behaviour_scores(near_ground)["hover_ticks"] == 0


def test_attitude_discipline_reports_worst_tilt_and_oscillation():
    """Oscillation is sign changes in angular velocity -- a controller
    fighting itself. Worst tilt was measured decisive in Arc 3 Phase B:
    every landing within 0.075 rad succeeded, every one past 0.139 failed."""
    steady = _steps([[1.0, -1.0, 0, 0, 0.05, 0.1, False, 0]] * 6)
    fighting = _steps([
        [1.0, -1.0, 0, 0, a, w, False, 0]
        for a, w in [(0.1, 0.5), (0.3, -0.5), (0.2, 0.5), (0.4, -0.5), (0.1, 0.5)]
    ])

    assert behaviour_scores(steady)["worst_tilt"] == pytest.approx(0.05)
    assert behaviour_scores(steady)["attitude_reversals"] == 0
    assert behaviour_scores(fighting)["worst_tilt"] == pytest.approx(0.4)
    assert behaviour_scores(fighting)["attitude_reversals"] == 4


def test_wasted_thrust_counts_burns_while_already_descending_safely():
    """Firing the main engine when already slower than the safe touchdown
    speed buys nothing -- it is fuel spent to hover."""
    # -0.08 normalised is 0.6 world u/s, comfortably under the 1.42 safe cap
    # but above the hover threshold, so this isolates wasted thrust from
    # dawdling. (-0.2 would be 1.5 u/s -- above the cap, so firing is earned.)
    wasteful = _steps([[1.0, -0.08, 0, 0, 0, 0, False, 2]] * 5)
    needed = _steps([[1.0, -5.0, 0, 0, 0, 0, False, 2]] * 5)

    assert behaviour_scores(wasteful)["wasted_main_frames"] == 5
    assert behaviour_scores(needed)["wasted_main_frames"] == 0


def test_trajectory_economy_penalises_undoing_your_own_drift():
    """Total lateral path travelled versus net displacement. A lander that
    drifts out and comes back spent time on motion that achieved nothing."""
    direct = _steps([[1.0, -1, x, 0, 0, 0, False, 0] for x in (0.0, 0.1, 0.2, 0.3)])
    wandering = _steps([[1.0, -1, x, 0, 0, 0, False, 0] for x in (0.0, 0.5, -0.4, 0.3)])

    assert behaviour_scores(direct)["lateral_waste"] == pytest.approx(0.0, abs=1e-9)
    assert behaviour_scores(wandering)["lateral_waste"] > 1.0


def test_grade_is_one_when_matching_the_ideal_and_falls_off_when_slower():
    """Grades are ideal/actual, so 1.0 means 'as fast as the model says is
    possible' and cannot exceed 1.0 -- beating the idealized plan would mean
    the plan is wrong, which is a finding, not a good grade."""
    assert grade(actual_ticks=100, ideal_ticks=100) == pytest.approx(1.0)
    assert grade(actual_ticks=200, ideal_ticks=100) == pytest.approx(0.5)
    assert grade(actual_ticks=50, ideal_ticks=100) == pytest.approx(1.0)


def test_grade_handles_a_segment_the_ideal_never_enters():
    """The idealized plan may skip a band entirely. Grading against zero
    would divide by zero or hand out a free 1.0."""
    assert np.isnan(grade(actual_ticks=40, ideal_ticks=0))
    assert np.isnan(grade(actual_ticks=0, ideal_ticks=0))
