import pytest

from lunar_lander_lab.controllers.heuristic import HeuristicController

# observation = (x, y, vx, vy, angle, angular_vel, leg1, leg2)
_LEVEL_DESCENT = (0.0, 1.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0)


def _obs(**overrides):
    names = ("x", "y", "vx", "vy", "angle", "angular_vel", "leg1", "leg2")
    values = dict(zip(names, _LEVEL_DESCENT))
    values.update(overrides)
    return tuple(values[n] for n in names)


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
        _obs(leg1=1.0, leg2=1.0),
    ],
)
def test_action_is_always_a_valid_discrete_action(obs):
    assert HeuristicController().get_action(obs) in {0, 1, 2, 3}


def test_both_legs_down_cuts_the_engines():
    """Firing after touchdown wastes fuel and can bounce the lander."""
    controller = HeuristicController()
    # Conditions that would otherwise demand a hard main-engine burn.
    assert controller.get_action(_obs(vy=-5.0, leg1=1.0, leg2=1.0)) == 0


def test_one_leg_down_does_not_cut_the_engines():
    controller = HeuristicController()
    assert controller.get_action(_obs(vy=-5.0, leg1=1.0, leg2=0.0)) != 0


def test_falling_too_fast_fires_the_main_engine():
    assert HeuristicController().get_action(_obs(vy=-5.0)) == 2


def test_tilt_fires_a_side_engine_before_the_main_engine():
    controller = HeuristicController()
    # Large angular error dominates, even while descending fast.
    assert controller.get_action(_obs(vy=-5.0, angular_vel=5.0)) in {1, 3}
    assert controller.get_action(_obs(vy=-5.0, angular_vel=-5.0)) in {1, 3}


def test_opposite_tilts_fire_opposite_engines():
    controller = HeuristicController()
    left = controller.get_action(_obs(angular_vel=5.0))
    right = controller.get_action(_obs(angular_vel=-5.0))
    assert {left, right} == {1, 3}


def test_gain_overrides_via_setattr_change_behaviour():
    """pid_search tunes gains with setattr; if that stopped working the whole
    search would silently evaluate one identical controller n_samples times."""
    obs = _obs(vy=-0.6)
    eager = HeuristicController()
    eager.HOVER_THRESHOLD = -10.0  # fire on essentially any descent
    lazy = HeuristicController()
    lazy.HOVER_THRESHOLD = 10.0  # never worth firing

    assert eager.get_action(obs) == 2
    assert lazy.get_action(obs) == 0
