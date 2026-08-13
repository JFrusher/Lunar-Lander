"""LQR: continuous state-feedback control from a Riccati solve.

A fourth classical family, genuinely different from the bang-bang heuristic,
gain-scheduled heuristic, and sampled-plan MPC controllers already here: a
fixed gain matrix K computed once, applied every step as u = u0 - K @ error.

Uses a continuous action space (Box(-1, 1, (2,))) -- LQR's control law is a
real-valued vector, and forcing it through discrete on/off actions would
mean re-deriving a bang-bang law LQR was never solving for. `env_kwargs`
(see base.py) tells `run`/`mark` to build the matching continuous env
automatically (Arc 4 Phase B).
"""

from typing import Optional, Sequence

import numpy as np
from gymnasium.envs.box2d.lunar_lander import FPS, SCALE, VIEWPORT_H, VIEWPORT_W
from scipy.linalg import solve_continuous_are

from ..utils.speed_ceiling import PlanarConstants, measure_planar_constants
from .base import BaseController
from .heuristic import _GAINS
from .registry import register_controller

# Same normalized-observation -> world-unit scaling LunarLander's own step()
# applies (see MPCController.obs_to_world for the discrete-action twin of
# this). Kept local rather than imported so importing this module never
# reaches controllers.mpc, which the package deliberately keeps out of eager
# import -- constructing an MPCController measures live-env constants, and
# importing controllers/__init__ should stay free of that cost.
_X_SCALE = VIEWPORT_W / SCALE / 2
_Y_SCALE = VIEWPORT_H / SCALE / 2
_VX_SCALE = FPS / _X_SCALE
_VY_SCALE = FPS / _Y_SCALE
_OMEGA_SCALE = FPS / 20.0

# The shipped heuristic's own TARGET_DESCENT_SPEED gain, converted from
# normalized to world units/s via the scaling above -- reused rather than
# re-picked so LQR targets the same safe descent rate the tuned controller
# already validated, and stays current if that gain is ever retuned again.
DEFAULT_TARGET_DESCENT_SPEED = abs(_GAINS["TARGET_DESCENT_SPEED"]) * _VY_SCALE


def _obs_to_state(observation: Sequence[float]) -> np.ndarray:
    """`[x, vx, vy, theta, omega]` in world units. `y` itself is omitted --
    it drives no other state and carries no cost term (see class docstring),
    so it plays no part in the regulator."""
    x, _y, vx, vy, theta, omega = observation[:6]
    return np.array([x * _X_SCALE, vx * _VX_SCALE, vy * _VY_SCALE, theta, omega * _OMEGA_SCALE])


class LQRController(BaseController):
    """Regulates toward level flight at a constant descent rate.

    Nonlinear model (matching `MPCController.rollout`'s physics, but with the
    main engine's real continuous throttle curve -- power = `0.5 +
    0.5*u_main` for `u_main > 0`, not linear from zero): the main engine
    thrusts along the hull axis, `(-sin theta, cos theta) * a_main_full *
    (0.5 + 0.5*u_main)`; the side thruster gives both a lateral kick
    (`a_lat * u_lat`) and a torque (`a_omega * u_lat`), linear in `u_lat`
    with no such offset. Linearizing at `theta=0, omega=0, u_lat=0,
    u_main=u0` (the trim throttle that exactly cancels gravity on that
    curve) gives a standard planar-VTOL model:

        d/dt [x, vx, vy, theta, omega] = A @ state + B @ [d(u_main), u_lat]

    with the only off-diagonal coupling `A[vx, theta] = gravity` (tilting
    couples to lateral accel) and `B` rows `[vx <- u_lat, vy <- d(u_main),
    omega <- u_lat]` -- the same "tilt to translate" mechanism Arc 3 Phase A
    found the fast idealized descent uses, here arising for free from the
    linearization rather than being searched for. `y` (altitude) is left out
    of the state entirely: it drives no other state in this model and there
    is no altitude target to track (only a descent *rate*), so including it
    would add an unobservable, undamped mode to the Riccati solve for no
    benefit.

    Targets a constant descent rate the whole way down -- one fixed gain
    matrix, not a schedule -- so expect a gentler, slower landing than the
    tuned/scheduled controllers. This exists to demonstrate a genuinely
    different control family (Riccati state feedback), not to win on speed;
    Arc 3's frontier already has its winners.
    """

    def __init__(
        self,
        const: Optional[PlanarConstants] = None,
        target_descent_speed: float = DEFAULT_TARGET_DESCENT_SPEED,
        q_diag: Sequence[float] = (0.5, 1.0, 2.0, 50.0, 15.0),
        r_diag: Sequence[float] = (1.0, 1.0),
    ):
        self.const = const if const is not None else measure_planar_constants()
        self.target_descent_speed = target_descent_speed

        dt = self.const.descent.dt
        gravity = self.const.descent.gravity  # negative, world units/s^2
        # `engine_dv_per_tick` is measured from the *discrete* action (see
        # measure_planar_constants), i.e. full throttle. LunarLander's own
        # step() maps a continuous action[0] to power via
        # `0.5 + 0.5*action0` for action0 > 0 (verified against Gymnasium's
        # source, gymnasium/envs/box2d/lunar_lander.py) -- not linearly from
        # zero. Fold that remap in here rather than at every call site.
        a_main_full = self.const.descent.engine_dv_per_tick / dt
        a_main = 0.5 * a_main_full  # d(accel)/d(u_main) on the "on" branch
        a_lat = self.const.side_dv_per_tick / dt
        a_omega = self.const.side_domega_per_tick / dt

        # Trim throttle: the constant command solving
        # 0.5*a_main_full*(1 + u0) + gravity = 0, i.e. the "on"-branch power
        # that exactly cancels gravity at theta=0.
        self.u0_main = -2.0 * gravity / a_main_full - 1.0

        # State order: [x, vx, vy, theta, omega]. Input order: [d(u_main), u_lat].
        A = np.zeros((5, 5))
        A[0, 1] = 1.0  # xdot = vx
        A[1, 3] = gravity  # d(vxdot)/dtheta at trim
        A[3, 4] = 1.0  # thetadot = omega

        B = np.zeros((5, 2))
        B[1, 1] = a_lat  # d(vxdot)/d(u_lat)
        B[2, 0] = a_main  # d(vydot)/d(u_main)
        # Negative: verified against the live env (action[1]=+1 measurably
        # *decreases* omega), matching MPCController.rollout's LEFT/RIGHT
        # convention. Getting this sign wrong turns the omega/theta loop
        # into positive feedback -- LQR chases its own tail and spins out,
        # regardless of Q/R tuning (found by tracing a live episode: omega
        # grew monotonically from t=0 under every gain tried).
        B[4, 1] = -a_omega  # d(omegadot)/d(u_lat)

        Q = np.diag(q_diag)
        R = np.diag(r_diag)
        P = solve_continuous_are(A, B, Q, R)
        self.K = np.linalg.solve(R, B.T @ P)

    def get_action(self, observation: Sequence[float]) -> np.ndarray:
        if observation[6] and observation[7]:
            return np.array([-1.0, 0.0], dtype=np.float32)  # both legs down: engines off

        state = _obs_to_state(observation)
        reference = np.array([0.0, 0.0, -self.target_descent_speed, 0.0, 0.0])
        error = state - reference
        du_main, u_lat = -self.K @ error

        u_main = np.clip(self.u0_main + du_main, -1.0, 1.0)
        u_lat = np.clip(u_lat, -1.0, 1.0)
        return np.array([u_main, u_lat], dtype=np.float32)

    @property
    def env_kwargs(self) -> dict:
        return {"continuous": True}


@register_controller("lqr")
def _build_lqr(model_name=None, gains_path=None) -> LQRController:
    return LQRController()
