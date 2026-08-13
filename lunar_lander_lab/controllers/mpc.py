"""Model-predictive control: plan a horizon, fly one step, replan.

Every other controller in this repo is reactive -- a proportional rule, or a
policy network mapping state to action with no lookahead. Arc 3 Phase A found
that the fast idealized landing *deliberately tilts* ~0.19 rad, both to
translate on the powerful main engine instead of the weak side ones and to
soften the effective vertical thrust so it tracks the touchdown-speed cap
more tightly. That is a planned manoeuvre, and a planner is the natural thing
to find it.

Plans with the cross-entropy method against the Phase A planar model:
sample action sequences, score them, refit the sampling distribution to the
best, repeat a couple of times, execute the first action of the winner.

Discrete actions, matching every other result in the repo -- and Arc 2 Phase
8b showed continuous control doubles the settling tail by feathering the
touchdown, which this sidesteps.
"""

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from gymnasium.envs.box2d.lunar_lander import FPS, SCALE, VIEWPORT_H, VIEWPORT_W

from ..utils.speed_ceiling import PlanarConstants, measure_planar_constants
from .base import BaseController
from .registry import register_controller

# The env reports a normalised observation; the planar model works in world
# units. See LunarLander's step(): x is divided by VIEWPORT_W/SCALE/2, vx is
# multiplied by that over FPS, and so on.
_X_SCALE = VIEWPORT_W / SCALE / 2
_Y_SCALE = VIEWPORT_H / SCALE / 2
_VX_SCALE = FPS / _X_SCALE
_VY_SCALE = FPS / _Y_SCALE
_OMEGA_SCALE = FPS / 20.0

N_ACTIONS = 4
_MAIN, _LEFT, _RIGHT = 2, 1, 3


def obs_to_world(observation: Sequence[float]) -> Dict[str, float]:
    """Convert a normalised env observation into world-unit model state."""
    return {
        "x": float(observation[0]) * _X_SCALE,
        "y": float(observation[1]) * _Y_SCALE,
        "vx": float(observation[2]) * _VX_SCALE,
        "vy": float(observation[3]) * _VY_SCALE,
        "theta": float(observation[4]),
        "omega": float(observation[5]) * _OMEGA_SCALE,
    }


def rollout(
    actions: np.ndarray, start: Dict[str, float], const: PlanarConstants
) -> Dict[str, np.ndarray]:
    """Roll `actions` (shape `(n_plans, horizon)`) forward, all plans at once.

    Vectorised because the scalar version is unaffordable: a few hundred plans
    over a ~25-step horizon, replanned every control step, is tens of millions
    of operations per evaluation run. Returns final state plus the tick each
    plan first reached the ground (`horizon` if it never did).
    """
    n_plans, horizon = actions.shape
    dt = const.descent.dt
    gravity_dv = const.descent.gravity * dt

    x = np.full(n_plans, start["x"], dtype=float)
    y = np.full(n_plans, start["y"], dtype=float)
    vx = np.full(n_plans, start["vx"], dtype=float)
    vy = np.full(n_plans, start["vy"], dtype=float)
    theta = np.full(n_plans, start["theta"], dtype=float)
    omega = np.full(n_plans, start["omega"], dtype=float)

    touchdown = np.full(n_plans, horizon, dtype=int)
    landed = np.zeros(n_plans, dtype=bool)
    touchdown_vy = np.zeros(n_plans, dtype=float)
    max_abs_theta = np.abs(np.full(n_plans, start["theta"], dtype=float))

    for t in range(horizon):
        a = actions[:, t]
        alive = ~landed

        main = (a == _MAIN) & alive
        left = (a == _LEFT) & alive
        right = (a == _RIGHT) & alive

        # Main engine thrusts along the hull axis: (-sin theta, cos theta).
        dv = const.descent.engine_dv_per_tick
        vx += np.where(main, -np.sin(theta) * dv, 0.0)
        vy += np.where(main, np.cos(theta) * dv, 0.0)

        # Side engines: torque plus a small lateral kick, opposite signs.
        omega += np.where(left, const.side_domega_per_tick, 0.0)
        omega -= np.where(right, const.side_domega_per_tick, 0.0)
        vx -= np.where(left, const.side_dv_per_tick, 0.0)
        vx += np.where(right, const.side_dv_per_tick, 0.0)

        vy += np.where(alive, gravity_dv, 0.0)

        theta += np.where(alive, omega * dt, 0.0)
        x += np.where(alive, vx * dt, 0.0)
        y += np.where(alive, vy * dt, 0.0)

        # Worst tilt reached while still flying. Tipping is a path constraint:
        # a plan can end level having rolled through 90 degrees on the way, by
        # which point the real lander has already crashed.
        max_abs_theta = np.where(alive, np.maximum(max_abs_theta, np.abs(theta)), max_abs_theta)

        newly = alive & (y <= 0.0)
        touchdown = np.where(newly, t + 1, touchdown)
        touchdown_vy = np.where(newly, vy, touchdown_vy)
        landed |= newly

    return {
        "x": x, "y": y, "vx": vx, "vy": vy, "theta": theta, "omega": omega,
        "touchdown": touchdown, "landed": landed, "touchdown_vy": touchdown_vy,
        "max_abs_theta": max_abs_theta,
    }


def scalar_rollout_reference(
    actions: List[int], start: Dict[str, float], const: PlanarConstants
) -> Dict[str, float]:
    """Plain-Python twin of `rollout`, for one plan. Exists only so a test can
    check the vectorised version against something obviously correct."""
    dt = const.descent.dt
    s = dict(start)
    for a in actions:
        if s["y"] <= 0.0:
            break
        if a == _MAIN:
            s["vx"] += -np.sin(s["theta"]) * const.descent.engine_dv_per_tick
            s["vy"] += np.cos(s["theta"]) * const.descent.engine_dv_per_tick
        elif a == _LEFT:
            s["omega"] += const.side_domega_per_tick
            s["vx"] -= const.side_dv_per_tick
        elif a == _RIGHT:
            s["omega"] -= const.side_domega_per_tick
            s["vx"] += const.side_dv_per_tick
        s["vy"] += const.descent.gravity * dt
        s["theta"] += s["omega"] * dt
        s["x"] += s["vx"] * dt
        s["y"] += s["vy"] * dt
    return s


def refit_elites(
    actions: np.ndarray,
    costs: np.ndarray,
    n_elites: int,
    n_actions: int = N_ACTIONS,
    smoothing: float = 0.05,
) -> np.ndarray:
    """Refit a per-timestep categorical distribution to the cheapest plans.

    `smoothing` keeps every action reachable. Zero probability is absorbing:
    an action ruled out on one iteration can never be sampled again, which is
    how CEM collapses early onto a mediocre plan.
    """
    horizon = actions.shape[1]
    elites = actions[np.argsort(costs)[:n_elites]]

    probs = np.empty((horizon, n_actions))
    for t in range(horizon):
        counts = np.bincount(elites[:, t], minlength=n_actions).astype(float)
        counts = counts / counts.sum() if counts.sum() else np.ones(n_actions) / n_actions
        probs[t] = (1 - smoothing) * counts + smoothing / n_actions
        probs[t] /= probs[t].sum()
    return probs


class MPCController(BaseController):
    """Cross-entropy-method receding-horizon controller."""

    def __init__(
        self,
        const: Optional[PlanarConstants] = None,
        n_samples: int = 256,
        horizon: int = 25,
        n_iters: int = 3,
        elite_frac: float = 0.15,
        safe_touchdown_speed: float = 1.42,
        seed: int = 0,
        # Terminal penalties, in "equivalent ticks" so they trade against the
        # time objective on a common scale.
        w_speed: float = 60.0,
        w_x: float = 12.0,
        w_vx: float = 8.0,
        w_theta: float = 4000.0,
        w_omega: float = 400.0,
        level_margin: float = 0.05,
        w_tip: float = 150.0,
        tip_margin: float = 0.35,
        w_no_return: float = 200.0,
        brake_margin: float = 1.6,
    ):
        self.const = const if const is not None else measure_planar_constants()
        self.n_samples = n_samples
        self.horizon = horizon
        self.n_iters = n_iters
        self.n_elites = max(2, int(n_samples * elite_frac))
        self.safe_touchdown_speed = safe_touchdown_speed
        self.rng = np.random.default_rng(seed)
        self.w_speed = w_speed
        self.w_x = w_x
        self.w_vx = w_vx
        self.w_theta = w_theta
        self.w_omega = w_omega
        self.level_margin = level_margin
        self.w_tip = w_tip
        self.tip_margin = tip_margin
        self.w_no_return = w_no_return
        self.brake_margin = brake_margin
        self._probs: Optional[np.ndarray] = None

    def cost_to_go(self, y: np.ndarray, vy: np.ndarray) -> np.ndarray:
        """Estimated ticks still needed to land safely from `(y, vy)`.

        Without this the planner dives. The horizon (~25 ticks) is far shorter
        than a descent (~190), so early on *no* sampled plan reaches the
        ground; scored on altitude lost, the cheapest plan is whichever falls
        fastest, and the arrival it is committing to lies beyond the horizon.

        The estimate is the same two-phase descent Phase A solves, in closed
        form so it can be evaluated for every plan at every step: coast at the
        current speed, then brake at the net rate to reach the touchdown cap.
        A plan whose braking distance already exceeds its remaining altitude
        is past the point of no return and is charged for it.
        """
        dt = self.const.descent.dt
        net_brake = self.const.descent.engine_dv_per_tick + self.const.descent.gravity * dt
        speed = np.maximum(-vy, 0.0)
        y = np.maximum(y, 0.0)

        excess = np.maximum(0.0, speed - self.safe_touchdown_speed)
        brake_ticks = excess / max(net_brake, 1e-6)
        # Margin, because the model's braking rate is optimistic: it assumes
        # the engine points straight down and is dedicated to braking, while
        # the real lander is also spending thrust and tilt on attitude, which
        # costs it a cos(theta) factor. Without this the planner always
        # intends to brake "later" -- and because it replans every step,
        # later never arrives and it arrives at 3-5 u/s instead of 1.4.
        brake_distance = (
            0.5 * (speed + self.safe_touchdown_speed) * brake_ticks * dt * self.brake_margin
        )

        # Remaining coast, floored so a hovering plan (speed ~ 0) yields a
        # large-but-finite estimate instead of dividing by zero.
        coast_distance = np.maximum(0.0, y - brake_distance)
        coast_ticks = coast_distance / np.maximum(speed, self.safe_touchdown_speed) / dt

        shortfall = np.maximum(0.0, brake_distance - y)
        return brake_ticks + coast_ticks + self.w_no_return * shortfall

    def _cost(self, out: Dict[str, np.ndarray]) -> np.ndarray:
        """Ticks to touchdown, plus estimated ticks still to come, plus what
        the arrival state gets wrong."""
        landed = out["landed"]
        elapsed = np.where(landed, out["touchdown"], self.horizon)
        remaining = np.where(landed, 0.0, self.cost_to_go(out["y"], out["vy"]))
        cost = elapsed + remaining

        impact = np.where(landed, -out["touchdown_vy"], -out["vy"])
        cost = cost + self.w_speed * np.maximum(0.0, impact - self.safe_touchdown_speed) ** 2
        # Where the lander will *end up*, not where it is. Within a 25-tick
        # horizon x barely moves, so penalising current x gives the planner no
        # gradient at all -- every plan scores the same and it drifts off-pad
        # (measured 4-7 world units out). Projecting the drift forward over
        # the remaining descent restores the gradient: shedding vx now visibly
        # reduces the miss.
        projected_x = out["x"] + out["vx"] * remaining * self.const.descent.dt
        cost = cost + self.w_x * projected_x ** 2
        cost = cost + self.w_vx * out["vx"] ** 2
        # Attitude at touchdown decides the landing. Measured: every episode
        # resting within 0.075 rad succeeded, every one past 0.139 failed. The
        # model has no contact dynamics, so it cannot see that a tilted
        # arrival rotates further on impact -- the cost has to stand in for
        # that, and it has to be big enough to outweigh the ~110 ticks of
        # flight time the planner would otherwise trade away for it.
        cost = cost + self.w_theta * np.maximum(0.0, np.abs(out["theta"]) - self.level_margin) ** 2
        # Arriving level but spinning is not arriving level -- the hull keeps
        # rotating through contact and tips. Measured as the dominant failure
        # mode before this term existed.
        cost = cost + self.w_omega * out["omega"] ** 2
        # Tipping anywhere along the path, not just at the end.
        cost = cost + self.w_tip * np.maximum(0.0, out["max_abs_theta"] - self.tip_margin) ** 2
        return cost

    def get_action(self, observation: Sequence[float]) -> Union[int, np.ndarray]:
        if observation[6] or observation[7]:
            # Already down. The model has no contact dynamics, so planning
            # here is planning against physics that no longer apply.
            return 0

        start = obs_to_world(observation)

        # Warm start: last plan shifted forward one tick. Replanning from
        # scratch every step wastes most of the search on rediscovering the
        # plan already committed to.
        if self._probs is not None:
            probs = np.vstack([self._probs[1:], np.ones((1, N_ACTIONS)) / N_ACTIONS])
        else:
            probs = np.ones((self.horizon, N_ACTIONS)) / N_ACTIONS

        best_plan = None
        for _ in range(self.n_iters):
            actions = np.empty((self.n_samples, self.horizon), dtype=int)
            for t in range(self.horizon):
                actions[:, t] = self.rng.choice(N_ACTIONS, size=self.n_samples, p=probs[t])
            costs = self._cost(rollout(actions, start, self.const))
            probs = refit_elites(actions, costs, self.n_elites)
            best_plan = actions[int(np.argmin(costs))]

        self._probs = probs
        return int(best_plan[0])

    def reset(self) -> None:
        """Drop the warm start. Call between episodes -- otherwise the first
        plan of a new episode is seeded by the end of the previous one."""
        self._probs = None


@register_controller("mpc")
def _build_mpc(model_name=None, gains_path=None) -> MPCController:
    return MPCController()
