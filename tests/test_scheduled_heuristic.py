"""Arc 3 Phase C: the heuristic uses one gain set for the whole descent, but
Arc 2 Phase 6 showed flight and settling are different regimes."""

import pytest

from lunar_lander_lab.controllers.heuristic import HeuristicController
from lunar_lander_lab.controllers.scheduled_heuristic import (
    BAND_EDGES,
    SCHEDULED_PARAM_SPACE,
    ScheduledHeuristicController,
    band_index,
    scheduled_gain_names,
)

_HIGH = [0.0, 1.2, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0]
_LOW = [0.0, 0.05, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0]


def test_band_index_walks_the_edges_top_down():
    """Band 0 is the highest altitude. Anything at or below the last edge is
    the terminal band."""
    assert band_index(5.0) == 0
    assert band_index(BAND_EDGES[0] - 1e-9) == 1
    assert band_index(-1.0) == len(BAND_EDGES)


def test_band_index_is_exact_at_an_edge():
    """Boundary behaviour is pinned deliberately: an edge belongs to the band
    above it, so the bands partition altitude with no gap or overlap."""
    assert band_index(BAND_EDGES[0]) == 0


def test_every_scheduled_gain_is_a_real_attribute():
    """setattr accepts typos silently -- a misspelled gain would be searched
    over and then quietly ignored, which is the failure this pins down.
    Arc 1 has the same test for the flat controller."""
    controller = ScheduledHeuristicController(tuned=False)
    for name in scheduled_gain_names():
        assert hasattr(controller, name), name


def test_param_space_covers_every_band_and_core_gain():
    names = set(SCHEDULED_PARAM_SPACE)
    assert names == set(scheduled_gain_names())
    # 3 bands x the 5 core gains. The attitude gains stay fixed: Arc 2 Phase 3
    # measured a flat objective surface there.
    assert len(names) == (len(BAND_EDGES) + 1) * 5


def test_reduces_to_the_flat_controller_when_all_bands_match():
    """A scheduled controller holding one gain set in every band must behave
    exactly like the shipped flat one -- otherwise the schedule itself is
    changing behaviour, and any measured gain is confounded."""
    flat = HeuristicController()
    scheduled = ScheduledHeuristicController(tuned=False)

    for obs in (_HIGH, _LOW, [0.4, 0.6, -0.3, -0.9, 0.15, 0.2, 0.0, 0.0]):
        assert scheduled.get_action(obs) == flat.get_action(obs)


def test_per_band_gains_actually_change_behaviour_by_altitude():
    """The whole point of scheduling. Same relative state at two altitudes
    must be able to produce different actions."""
    scheduled = ScheduledHeuristicController(tuned=False)
    # Band 0 hovers readily, the terminal band almost never does.
    scheduled.B0_HOVER_THRESHOLD = -10.0
    scheduled.B2_HOVER_THRESHOLD = 10.0

    assert scheduled.get_action(_HIGH) != scheduled.get_action(_LOW)


def test_legs_down_still_cuts_the_engine():
    scheduled = ScheduledHeuristicController(tuned=False)

    assert scheduled.get_action([0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 1.0, 1.0]) == 0


@pytest.mark.parametrize("obs", [_HIGH, _LOW])
def test_actions_stay_in_the_valid_discrete_set(obs):
    assert ScheduledHeuristicController().get_action(obs) in (0, 1, 2, 3)


def test_ships_tuned_gains_that_differ_from_flat():
    """The tuned config is the point of the phase -- if it silently failed to
    load, the controller would quietly be the flat one wearing a new name."""
    tuned = ScheduledHeuristicController()
    untuned = ScheduledHeuristicController(tuned=False)

    assert tuned.B0_TARGET_DESCENT_SPEED != untuned.B0_TARGET_DESCENT_SPEED
    # Fall fast up high, slow on approach -- the profile the search found.
    assert tuned.B0_TARGET_DESCENT_SPEED < tuned.B2_TARGET_DESCENT_SPEED


def test_rejects_an_unknown_scheduled_gain():
    with pytest.raises(ValueError, match="unknown scheduled gain"):
        ScheduledHeuristicController(gains={"B9_NOPE": 1.0})
