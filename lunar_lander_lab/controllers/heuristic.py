"""Rule-based PID-style controller for LunarLander-v3."""

from typing import Sequence

from .base import BaseController


class HeuristicController(BaseController):
    """Hand-tuned proportional controller.

    Drives a target horizontal angle from horizontal position/velocity error,
    then converts angle/altitude error into engine firings. Gains are exposed
    as class attributes so they can be tweaked without touching the logic.
    """

    # Horizontal position -> desired angle (radians)
    ANGLE_GAIN_POS = 0.5
    ANGLE_GAIN_VEL = 1.0

    # Angle/angular-velocity -> desired angular action strength
    ANGLE_ERROR_GAIN = 0.5
    ANGULAR_VEL_GAIN = 1.0

    # Vertical descent speed target and gain
    TARGET_DESCENT_SPEED = -0.15
    DESCENT_GAIN = 0.5

    # Thresholds that decide when to fire an engine
    ANGLE_THRESHOLD = 0.05
    HOVER_THRESHOLD = 0.05

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
