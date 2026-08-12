"""Arc 3 Phase B: does planning beat reacting?

Every controller in this repo so far is reactive. MPC plans a horizon each
step against the Phase A planar model, executes the first action, replans.
"""

import numpy as np
import pytest

from lunar_lander_lab.controllers.mpc import (
    MPCController,
    obs_to_world,
    refit_elites,
    rollout,
    scalar_rollout_reference,
)
from lunar_lander_lab.utils.speed_ceiling import DescentConstants, PlanarConstants

CONST = PlanarConstants(
    descent=DescentConstants(mass=5.0, gravity=-10.0, engine_dv_per_tick=0.36, dt=0.02),
    side_dv_per_tick=0.05,
    side_domega_per_tick=0.095,
    inertia=0.83,
)


def test_obs_to_world_undoes_the_env_normalisation():
    """The env reports normalised state (x/10, y/6.67, vx/5, vy/7.5, angle,
    omega/2.5); the model works in world units. Mixing the two is the same
    class of bug that made touchdown speed read 7.5x too gentle in Arc 2."""
    obs = [0.5, 1.2, 0.4, -0.2, 0.1, 0.3, 0.0, 0.0]

    state = obs_to_world(obs)

    assert state["x"] == pytest.approx(5.0)
    assert state["y"] == pytest.approx(1.2 * (400 / 30 / 2))
    assert state["vx"] == pytest.approx(2.0)
    assert state["vy"] == pytest.approx(-1.5)
    assert state["theta"] == pytest.approx(0.1)
    assert state["omega"] == pytest.approx(0.75)


def test_vectorised_rollout_matches_a_scalar_reference():
    """The whole planner rests on the vectorised model. If it disagrees with
    a straightforward scalar implementation, every plan is scored against
    physics that don't exist."""
    rng = np.random.default_rng(0)
    actions = rng.integers(0, 4, size=(5, 12))
    start = {"x": 0.3, "y": 9.0, "vx": -1.5, "vy": -2.0, "theta": 0.05, "omega": -0.1}

    vec = rollout(actions, start, CONST)

    for i in range(actions.shape[0]):
        ref = scalar_rollout_reference(list(actions[i]), start, CONST)
        for key in ("x", "y", "vx", "vy", "theta", "omega"):
            assert vec[key][i] == pytest.approx(ref[key], rel=1e-9, abs=1e-9), key


def test_main_engine_thrust_follows_the_hull_angle():
    """Thrust direction is (-sin theta, cos theta): tilted right, the lander
    is pushed left. Getting this sign wrong makes the planner steer backwards."""
    start = {"x": 0.0, "y": 9.0, "vx": 0.0, "vy": 0.0, "theta": 0.4, "omega": 0.0}
    main = np.array([[2]])

    out = rollout(main, start, CONST)

    assert out["vx"][0] < 0, "positive tilt must push the lander in -x"
    assert out["vy"][0] > CONST.descent.gravity * CONST.descent.dt, "and still lift"


def test_noop_is_pure_free_fall():
    start = {"x": 0.0, "y": 9.0, "vx": 0.0, "vy": 0.0, "theta": 0.0, "omega": 0.0}

    out = rollout(np.array([[0, 0, 0]]), start, CONST)

    assert out["vy"][0] == pytest.approx(3 * CONST.descent.gravity * CONST.descent.dt)
    assert out["vx"][0] == pytest.approx(0.0)


def test_side_engines_rotate_in_opposite_directions():
    start = {"x": 0.0, "y": 9.0, "vx": 0.0, "vy": 0.0, "theta": 0.0, "omega": 0.0}

    left = rollout(np.array([[1]]), start, CONST)
    right = rollout(np.array([[3]]), start, CONST)

    assert left["omega"][0] > 0 > right["omega"][0]


def test_refit_elites_concentrates_on_what_the_elites_did():
    """CEM's whole mechanism. If every elite chose action 2 at t=0, the next
    sampling round must strongly prefer action 2 there."""
    actions = np.array([[2, 0], [2, 1], [2, 3]])
    costs = np.array([1.0, 2.0, 3.0])

    probs = refit_elites(actions, costs, n_elites=3, n_actions=4, smoothing=0.0)

    assert probs[0, 2] == pytest.approx(1.0)
    assert probs[0].sum() == pytest.approx(1.0)
    assert probs[1].sum() == pytest.approx(1.0)


def test_refit_elites_keeps_some_probability_everywhere_when_smoothed():
    """Zero probability is absorbing -- an action ruled out early can never be
    reconsidered, which is how CEM collapses onto a bad plan."""
    actions = np.array([[2, 2], [2, 2]])
    costs = np.array([1.0, 2.0])

    probs = refit_elites(actions, costs, n_elites=1, n_actions=4, smoothing=0.1)

    assert (probs > 0).all()
    assert probs.sum(axis=1) == pytest.approx(np.ones(2))


def test_refit_elites_selects_the_cheapest():
    actions = np.array([[0, 0], [3, 3], [1, 1]])
    costs = np.array([9.0, 1.0, 5.0])

    probs = refit_elites(actions, costs, n_elites=1, n_actions=4, smoothing=0.0)

    assert probs[0, 3] == pytest.approx(1.0)


def test_rollout_tracks_worst_tilt_along_the_path():
    """Tipping over is a path constraint, not a terminal one -- a plan can end
    level having rolled through 90 degrees on the way, by which point the real
    lander has already crashed."""
    start = {"x": 0.0, "y": 9.0, "vx": 0.0, "vy": 0.0, "theta": 0.0, "omega": 3.0}

    out = rollout(np.array([[0, 0, 0, 0, 0]]), start, CONST)

    assert out["max_abs_theta"][0] > abs(out["theta"][0]) or out["max_abs_theta"][0] > 0.0
    assert out["max_abs_theta"][0] == pytest.approx(abs(out["theta"][0]))


def test_rollout_worst_tilt_catches_a_swing_that_returns_level():
    """Start tilted and rotating back through level: the plan ends nearly
    upright, but it was badly tilted for most of the path. Reporting only the
    final angle would call this plan safe."""
    start = {"x": 0.0, "y": 9.0, "vx": 0.0, "vy": 0.0, "theta": 0.5, "omega": -5.0}

    out = rollout(np.array([[0] * 6]), start, CONST)

    assert abs(out["theta"][0]) < 0.2, "should have swung back near level"
    assert out["max_abs_theta"][0] == pytest.approx(0.5)


def test_cost_penalises_arriving_while_rotating():
    """Landing level but spinning is not landing level. Without an omega term
    the planner is indifferent to arriving in a spin."""
    controller = MPCController(const=CONST, n_samples=8, horizon=4, n_iters=1, seed=0)

    still = {k: np.array(v) for k, v in {
        "x": [0.0], "y": [0.0], "vx": [0.0], "vy": [-1.0], "theta": [0.0], "omega": [0.0],
        "touchdown": [4], "landed": [True], "touchdown_vy": [-1.0], "max_abs_theta": [0.0],
    }.items()}
    spinning = dict(still, omega=np.array([2.5]))

    assert controller._cost(spinning)[0] > controller._cost(still)[0]


def test_cost_penalises_tipping_along_the_path():
    controller = MPCController(const=CONST, n_samples=8, horizon=4, n_iters=1, seed=0)

    level = {k: np.array(v) for k, v in {
        "x": [0.0], "y": [0.0], "vx": [0.0], "vy": [-1.0], "theta": [0.0], "omega": [0.0],
        "touchdown": [4], "landed": [True], "touchdown_vy": [-1.0], "max_abs_theta": [0.05],
    }.items()}
    tipped = dict(level, max_abs_theta=np.array([1.5]))

    assert controller._cost(tipped)[0] > controller._cost(level)[0]


def test_cost_to_go_penalises_diving_past_the_point_of_no_return():
    """The failure this exists to prevent: with a horizon too short to reach
    the ground, a planner scored only on altitude lost will dive, because
    every metre of descent looks like progress and the arrival is beyond its
    sight. Cost-to-go prices the braking distance the dive is committing to."""
    controller = MPCController(const=CONST, n_samples=8, horizon=5, n_iters=1, seed=0)

    # Same altitude, wildly different descent rates.
    gentle = {"y": np.array([4.0]), "vy": np.array([-1.0])}
    diving = {"y": np.array([4.0]), "vy": np.array([-9.0])}

    assert controller.cost_to_go(diving["y"], diving["vy"]) > controller.cost_to_go(
        gentle["y"], gentle["vy"]
    )


def test_cost_to_go_is_finite_at_zero_velocity():
    """A hovering plan has vy ~ 0; a naive distance/speed estimate divides by
    zero and poisons the whole cost vector with inf or nan."""
    controller = MPCController(const=CONST, n_samples=8, horizon=5, n_iters=1, seed=0)

    cost = controller.cost_to_go(np.array([5.0]), np.array([0.0]))

    assert np.isfinite(cost).all()


def test_cost_to_go_is_zero_on_the_ground():
    controller = MPCController(const=CONST, n_samples=8, horizon=5, n_iters=1, seed=0)

    assert controller.cost_to_go(np.array([0.0]), np.array([-1.0]))[0] == pytest.approx(0.0)


@pytest.mark.slow
def test_controller_returns_a_valid_discrete_action():
    controller = MPCController(n_samples=32, horizon=10, n_iters=1, seed=0)

    for obs in (
        [0.0, 1.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.2, -0.3, -0.8, 0.2, 0.1, 0.0, 0.0],
        [-0.4, 1.4, 0.6, 0.1, -0.15, -0.2, 0.0, 0.0],
    ):
        action = controller.get_action(obs)
        assert action in (0, 1, 2, 3)


@pytest.mark.slow
def test_controller_coasts_when_high_and_slow():
    """Falling gently from altitude, burning fuel now is strictly wasteful --
    a planner that fires here has its cost function backwards."""
    controller = MPCController(n_samples=256, horizon=25, n_iters=3, seed=0)

    action = controller.get_action([0.0, 1.4, 0.0, -0.05, 0.0, 0.0, 0.0, 0.0])

    assert action != 2
