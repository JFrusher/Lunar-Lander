"""Rule-based PID-style controller for LunarLander-v3."""

import json
from pathlib import Path
from typing import Sequence

from .base import BaseController

_GAINS_PATH = Path(__file__).resolve().parent.parent / "configs" / "heuristic_gains.json"
_GAINS = json.loads(_GAINS_PATH.read_text())


class HeuristicController(BaseController):
    """Hand-tuned proportional controller.

    Drives a target horizontal angle from horizontal position/velocity error,
    then converts angle/altitude error into engine firings. Gains are exposed
    as class attributes so they can be tweaked without touching the logic.
    """

    # Swept gains: configs/heuristic_gains.json (see utils/pid_search.py to re-tune)
    ANGLE_GAIN_VEL = _GAINS["ANGLE_GAIN_VEL"]
    DESCENT_GAIN = _GAINS["DESCENT_GAIN"]
    TARGET_DESCENT_SPEED = _GAINS["TARGET_DESCENT_SPEED"]
    ANGLE_THRESHOLD = _GAINS["ANGLE_THRESHOLD"]
    HOVER_THRESHOLD = _GAINS["HOVER_THRESHOLD"]

    # Never swept, held at original defaults:
    ANGLE_GAIN_POS = 0.5  # horizontal position -> desired angle
    ANGLE_ERROR_GAIN = 0.5  # angle error -> angular correction
    ANGULAR_VEL_GAIN = 1.0  # angular velocity damping

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
