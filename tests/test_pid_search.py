import json

import numpy as np
import pytest

from lunar_lander_lab.controllers.heuristic import HeuristicController
from lunar_lander_lab.utils.pid_search import (
    CORE_PARAM_SPACE,
    EXTENDED_PARAM_SPACE,
    sample_gain_sets,
)

_ATTITUDE_GAINS = ("ANGLE_GAIN_POS", "ANGLE_ERROR_GAIN", "ANGULAR_VEL_GAIN")


def test_extended_space_is_core_plus_three_attitude_gains():
    assert set(EXTENDED_PARAM_SPACE) == set(CORE_PARAM_SPACE) | set(_ATTITUDE_GAINS)
    for name, bounds in CORE_PARAM_SPACE.items():
        assert EXTENDED_PARAM_SPACE[name] == bounds, f"{name} bounds drifted"


@pytest.mark.parametrize("space", [CORE_PARAM_SPACE, EXTENDED_PARAM_SPACE])
def test_samples_have_right_shape_and_stay_in_bounds(space):
    samples = sample_gain_sets(50, space, seed=0)
    assert len(samples) == 50
    for gains in samples:
        assert set(gains) == set(space)
        for name, value in gains.items():
            low, high = space[name]
            assert low <= value <= high, (name, value)


@pytest.mark.parametrize("space", [CORE_PARAM_SPACE, EXTENDED_PARAM_SPACE])
def test_latin_hypercube_stratification(space):
    """Every 1-D projection puts exactly one sample in each of the n bins.

    This is the property that makes it a Latin Hypercube rather than plain
    uniform sampling, and the property the fixed-seed bug (commit 9b6729a)
    quietly relied on being reproducible.
    """
    n = 40
    samples = sample_gain_sets(n, space, seed=3)
    for name, (low, high) in space.items():
        unit = np.array([(g[name] - low) / (high - low) for g in samples])
        bins = np.floor(unit * n).astype(int)
        assert sorted(bins) == list(range(n)), name


def test_sampling_is_seed_reproducible():
    assert sample_gain_sets(10, EXTENDED_PARAM_SPACE, seed=7) == sample_gain_sets(
        10, EXTENDED_PARAM_SPACE, seed=7
    )
    assert sample_gain_sets(10, EXTENDED_PARAM_SPACE, seed=7) != sample_gain_sets(
        10, EXTENDED_PARAM_SPACE, seed=8
    )


def test_every_extended_gain_is_a_real_controller_attribute():
    """pid_search tunes gains via setattr, which silently accepts typos."""
    controller = HeuristicController()
    for name in EXTENDED_PARAM_SPACE:
        assert hasattr(controller, name), name


def test_all_gains_load_from_config():
    """The 3 attitude gains are config-loaded now, not hardcoded."""
    from lunar_lander_lab.controllers.heuristic import _GAINS_PATH

    shipped = json.loads(_GAINS_PATH.read_text())
    controller = HeuristicController()
    for name in EXTENDED_PARAM_SPACE:
        assert name in shipped, f"{name} missing from heuristic_gains.json"
        assert getattr(controller, name) == shipped[name], name
