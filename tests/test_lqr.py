"""Arc 4 Phase C: a fourth classical family (Riccati state feedback),
genuinely different from the bang-bang heuristic/gain-scheduled/MPC
controllers already here."""

import numpy as np
import pytest

from lunar_lander_lab.controllers.lqr import LQRController
from lunar_lander_lab.utils.speed_ceiling import DescentConstants, PlanarConstants

# Same fixed constants test_mpc.py uses -- no live env, so this stays fast.
CONST = PlanarConstants(
    descent=DescentConstants(mass=5.0, gravity=-10.0, engine_dv_per_tick=0.36, dt=0.02),
    side_dv_per_tick=0.05,
    side_domega_per_tick=0.095,
    inertia=0.83,
)

_LEVEL_DESCENT = (0.0, 1.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0)


def _obs(**overrides):
    names = ("x", "y", "vx", "vy", "angle", "angular_vel", "leg1", "leg2")
    values = dict(zip(names, _LEVEL_DESCENT))
    values.update(overrides)
    return tuple(values[n] for n in names)


def test_trim_throttle_is_a_valid_action_value():
    """u0_main solves 0.5*a_main_full*(1+u0) + gravity = 0 -- must land inside
    (-1, 1] (a valid, "on"-branch main-engine action) or every other gain in
    the model is being computed around a nonsensical operating point."""
    controller = LQRController(const=CONST)
    assert -1.0 < controller.u0_main <= 1.0


def test_gain_matrix_is_finite_and_correctly_shaped():
    controller = LQRController(const=CONST)
    assert controller.K.shape == (2, 5)
    assert np.all(np.isfinite(controller.K))


def test_closed_loop_is_stable():
    """The point of solving a Riccati equation instead of guessing gains:
    A - B@K must have every eigenvalue in the open left half-plane. If this
    fails, the "regulator" actively drives the state away from equilibrium."""
    controller = LQRController(const=CONST)
    dt = CONST.descent.dt
    gravity = CONST.descent.gravity
    a_main_full = CONST.descent.engine_dv_per_tick / dt
    a_lat = CONST.side_dv_per_tick / dt
    a_omega = CONST.side_domega_per_tick / dt

    A = np.zeros((5, 5))
    A[0, 1] = 1.0
    A[1, 3] = gravity
    A[3, 4] = 1.0
    B = np.zeros((5, 2))
    B[1, 1] = a_lat
    B[2, 0] = 0.5 * a_main_full
    B[4, 1] = -a_omega

    eigenvalues = np.linalg.eigvals(A - B @ controller.K)
    assert np.all(eigenvalues.real < 0)


def test_engines_cut_when_both_legs_are_down():
    controller = LQRController(const=CONST)
    action = controller.get_action(_obs(leg1=1.0, leg2=1.0))
    assert action[0] <= 0.0  # main engine off
    assert action[1] == 0.0  # side thruster neutral


@pytest.mark.parametrize(
    "obs",
    [
        _obs(),
        _obs(x=1.0, vx=1.0),
        _obs(x=-1.0, vx=-1.0),
        _obs(angle=0.5, angular_vel=2.0),
        _obs(angle=-0.5, angular_vel=-2.0),
        _obs(vy=-5.0),
        _obs(vy=5.0),
    ],
)
def test_action_is_always_within_bounds(obs):
    controller = LQRController(const=CONST)
    action = controller.get_action(obs)
    assert action.shape == (2,)
    assert -1.0 <= action[0] <= 1.0
    assert -1.0 <= action[1] <= 1.0


def test_env_kwargs_is_continuous():
    assert LQRController(const=CONST).env_kwargs == {"continuous": True}


@pytest.mark.slow
def test_lands_without_crashing_over_held_out_episodes():
    """End-to-end against the real env. Regression guard for the two bugs
    found while building this: a wrong main-engine trim/slope (the throttle
    curve is `0.5 + 0.5*action`, not linear from zero) and an inverted
    u_lat -> omega sign (verified against the live env: action[1]=+1
    measurably *decreases* omega) -- either one alone made every episode
    crash. Held-out seeds (matching pid_search.HOLDOUT_SEED_START), not the
    range used while debugging."""
    from lunar_lander_lab.utils.evaluation import evaluate_controller_natural
    from lunar_lander_lab.utils.pid_search import HOLDOUT_SEED_START

    controller = LQRController()
    metrics = evaluate_controller_natural(
        controller, num_episodes=15, seed_start=HOLDOUT_SEED_START, env_kwargs=controller.env_kwargs
    )
    assert metrics["crash_rate_pct"] == 0.0
    assert metrics["success_rate_pct"] >= 50.0
