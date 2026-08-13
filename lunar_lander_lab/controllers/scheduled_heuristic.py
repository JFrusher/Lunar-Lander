"""Altitude-scheduled gains for the proportional controller.

`HeuristicController` applies one gain set across the whole descent. Arc 2
Phase 6 split every episode at first ground contact and found flight and
settling are different regimes with different dynamics -- so a single set of
gains is being asked to do two jobs. This splits them by altitude.

Only the 5 core gains are scheduled. The 3 attitude gains stay fixed at the
shipped values: Arc 2 Phase 3 swept them and measured a flat objective
surface, so scheduling them would triple the search dimensionality for
nothing.

Gains are exposed as flat `B<band>_<GAIN>` attributes rather than a nested
structure, so `pid_search.sample_gain_sets()` and `_evaluate_gain_set()`'s
existing `setattr` mechanism work on this controller unchanged.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .heuristic import _GAINS, HeuristicController
from .registry import register_controller

_SCHEDULED_GAINS_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "scheduled_gains.json"
)

# Normalised altitude (observation[1]) boundaries, high to low. Band 0 is the
# initial descent, band 1 the approach, band 2 the terminal phase. Chosen so
# band 2 covers roughly the last stretch before contact, where Arc 2 Phase 6
# located the regime change.
BAND_EDGES: Tuple[float, ...] = (0.6, 0.15)

SCHEDULED_GAINS: Tuple[str, ...] = (
    "ANGLE_GAIN_VEL",
    "DESCENT_GAIN",
    "TARGET_DESCENT_SPEED",
    "ANGLE_THRESHOLD",
    "HOVER_THRESHOLD",
)


def band_index(altitude: float) -> int:
    """Which band an altitude falls in. Edges belong to the band above, so the
    bands partition altitude with no gap and no overlap."""
    for i, edge in enumerate(BAND_EDGES):
        if altitude >= edge:
            return i
    return len(BAND_EDGES)


def scheduled_gain_names() -> List[str]:
    return [
        f"B{band}_{gain}"
        for band in range(len(BAND_EDGES) + 1)
        for gain in SCHEDULED_GAINS
    ]


def _band_space() -> Dict[str, Tuple[float, float]]:
    """Per-band bounds, reusing the audited flat ranges.

    Arc 2 Phase 2b checked these bounds against real data and found no edge
    pressure, so they are reused rather than re-derived -- scheduling changes
    where a gain applies, not what values are plausible.
    """
    from ..utils.pid_search import CORE_PARAM_SPACE

    return {
        f"B{band}_{gain}": CORE_PARAM_SPACE[gain]
        for band in range(len(BAND_EDGES) + 1)
        for gain in SCHEDULED_GAINS
    }


SCHEDULED_PARAM_SPACE: Dict[str, Tuple[float, float]] = _band_space()


class ScheduledHeuristicController(HeuristicController):
    """Proportional controller with per-altitude-band core gains.

    Defaults to the shipped flat gains in every band, so an untuned instance
    is behaviourally identical to `HeuristicController` -- which makes any
    measured difference attributable to the schedule rather than to the
    scheduling machinery.

    Hard switch at band edges. The discontinuity is a real risk (a gain jump
    mid-descent can kick the controller), and linear interpolation between
    bands is the fallback if boundary behaviour looks bad.
    """

    def __init__(self, gains: Optional[Dict[str, float]] = None, tuned: bool = True) -> None:
        # Start from the flat gains in every band, so an untuned instance is
        # behaviourally identical to HeuristicController.
        for band in range(len(BAND_EDGES) + 1):
            for gain in SCHEDULED_GAINS:
                setattr(self, f"B{band}_{gain}", _GAINS[gain])

        if gains is None and tuned and _SCHEDULED_GAINS_PATH.exists():
            gains = json.loads(_SCHEDULED_GAINS_PATH.read_text())
        for name, value in (gains or {}).items():
            if not hasattr(self, name):
                raise ValueError(f"unknown scheduled gain {name!r}")
            setattr(self, name, value)

    def get_action(self, observation: Sequence[float]) -> int:
        band = band_index(float(observation[1]))
        # Point the base class's gain lookups at this band, then defer to it
        # entirely -- the control law is unchanged and stays in one place.
        for gain in SCHEDULED_GAINS:
            setattr(self, gain, getattr(self, f"B{band}_{gain}"))
        return super().get_action(observation)


@register_controller("scheduled")
def _build_scheduled(model_name=None, gains_path=None) -> ScheduledHeuristicController:
    gains = json.loads(Path(gains_path).read_text()) if gains_path else None
    return ScheduledHeuristicController(gains=gains)
