from lunar_lander_lab.utils.lockout_sweep import (
    ALTITUDE_THRESHOLDS,
    STEP_THRESHOLDS,
    gate_label,
    lockout_grid,
)


def test_lockout_grid_is_step_thresholds_plus_altitude_thresholds():
    """Not a cross product -- the two gate forms are alternatives being
    compared, not combined."""
    grid = lockout_grid()

    assert grid == (
        [{"lockout_steps": s} for s in STEP_THRESHOLDS]
        + [{"altitude_threshold": a} for a in ALTITUDE_THRESHOLDS]
    )


def test_lockout_grid_accepts_explicit_thresholds():
    """The second pass re-runs step gates on a finer grid without paying to
    retrain the altitude gates, which already failed outright on both legs."""
    grid = lockout_grid(step_thresholds=[5, 10], altitude_thresholds=[])

    assert grid == [{"lockout_steps": 5}, {"lockout_steps": 10}]


def test_lockout_grid_step_thresholds_cover_the_viable_region():
    """20 won the first pass and 50 already broke, so the optimum -- if there
    is one -- lives below 50. A grid that only samples 20 there can't locate
    it."""
    below_50 = [s for s in STEP_THRESHOLDS if s < 50]

    assert len(below_50) >= 4, f"too coarse below the 50-step cliff: {below_50}"


def test_gate_label_is_unique_and_readable_per_config():
    labels = [gate_label(g) for g in lockout_grid()]

    assert len(labels) == len(set(labels)), "every grid config needs a distinct label"
    for s in STEP_THRESHOLDS:
        assert str(s) in gate_label({"lockout_steps": s})
    for a in ALTITUDE_THRESHOLDS:
        assert str(a) in gate_label({"altitude_threshold": a})
