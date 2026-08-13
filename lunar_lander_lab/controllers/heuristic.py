"""Rule-based PID-style controller for LunarLander-v3."""

import json
from pathlib import Path
from typing import Sequence

from .base import BaseController
from .registry import register_controller

_GAINS_PATH = Path(__file__).resolve().parent.parent / "configs" / "heuristic_gains.json"
_GAINS = json.loads(_GAINS_PATH.read_text())


class HeuristicController(BaseController):
    """Hand-tuned proportional controller.

    Drives a target horizontal angle from horizontal position/velocity error,
    then converts angle/altitude error into engine firings. Gains are exposed
    as class attributes so they can be tweaked without touching the logic.
    """

    # All gains: configs/heuristic_gains.json (see utils/pid_search.py to re-tune)
    ANGLE_GAIN_VEL = _GAINS["ANGLE_GAIN_VEL"]
    DESCENT_GAIN = _GAINS["DESCENT_GAIN"]
    TARGET_DESCENT_SPEED = _GAINS["TARGET_DESCENT_SPEED"]
    ANGLE_THRESHOLD = _GAINS["ANGLE_THRESHOLD"]
    HOVER_THRESHOLD = _GAINS["HOVER_THRESHOLD"]

    ANGLE_GAIN_POS = _GAINS["ANGLE_GAIN_POS"]  # horizontal position -> desired angle
    ANGLE_ERROR_GAIN = _GAINS["ANGLE_ERROR_GAIN"]  # angle error -> angular correction
    ANGULAR_VEL_GAIN = _GAINS["ANGULAR_VEL_GAIN"]  # angular velocity damping

    def get_action(self, observation: Sequence[float]) -> int:
        x, y, vx, vy, angle, angular_vel, leg1, leg2 = observation

        if leg1 and leg2:
            return 0

        target_angle = self.ANGLE_GAIN_POS * x + self.ANGLE_GAIN_VEL * vx
        target_angle = max(-0.4, min(0.4, target_angle))

        angle_error = target_angle - angle
        angular_correction = (
            self.ANGLE_ERROR_GAIN * angle_error - self.ANGULAR_VEL_GAIN * angular_vel
        )

        hover_error = self.TARGET_DESCENT_SPEED - vy
        vertical_correction = self.DESCENT_GAIN * hover_error

        if angular_correction < -self.ANGLE_THRESHOLD:
            return 3  # fire right engine, rotate left
        if angular_correction > self.ANGLE_THRESHOLD:
            return 1  # fire left engine, rotate right
        if vertical_correction > self.HOVER_THRESHOLD:
            return 2  # fire main engine
        return 0


@register_controller("heuristic")
def _build_heuristic(model_name=None, gains_path=None) -> HeuristicController:
    controller = HeuristicController()
    if gains_path:
        # Override the shipped gains so an older set can be flown
        # side-by-side with the current one. Same setattr mechanism
        # pid_search uses to evaluate a sampled gain set.
        for gain, value in json.loads(Path(gains_path).read_text()).items():
            if not hasattr(controller, gain):
                raise ValueError(f"{gains_path}: unknown gain {gain!r}")
            setattr(controller, gain, value)
    return controller
