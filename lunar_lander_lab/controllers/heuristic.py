"""Rule-based PID-style controller for LunarLander-v3."""

from typing import Sequence

from .base import BaseController


class HeuristicController(BaseController):
    """Hand-tuned proportional controller.

    Drives a target horizontal angle from horizontal position/velocity error,
    then converts angle/altitude error into engine firings. Gains are exposed
    as class attributes so they can be tweaked without touching the logic.
    """

    # Defaults from the Monte Carlo gain sweep (utils/pid_search.py), v2 run:
    # 500 Latin-Hypercube samples x 50 episodes, best-found set
    # (mean_reward 254.3, success 100%, crash 0%). See pid_search_results_v2/.

    # Horizontal position -> desired angle (radians)
    ANGLE_GAIN_POS = 0.5  # not swept, held at original default
    ANGLE_GAIN_VEL = 1.175571804391529

    # Angle/angular-velocity -> desired angular action strength
    ANGLE_ERROR_GAIN = 0.5  # not swept, held at original default
    ANGULAR_VEL_GAIN = 1.0  # not swept, held at original default

    # Vertical descent speed target and gain
    TARGET_DESCENT_SPEED = -0.10534496960720757
    DESCENT_GAIN = 1.3919415369112689

    # Thresholds that decide when to fire an engine
    ANGLE_THRESHOLD = 0.03654782964522024
    HOVER_THRESHOLD = 0.12111246024626458

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
