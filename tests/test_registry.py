"""Controller registry: build_controller()/controller_names() replace cli.py's
old hardcoded if/elif chain and duplicated choices=[...] lists."""

import pytest

from lunar_lander_lab.controllers import build_controller, controller_names
from lunar_lander_lab.controllers.heuristic import HeuristicController
from lunar_lander_lab.controllers.scheduled_heuristic import ScheduledHeuristicController


def test_controller_names_includes_mpc_without_importing_it():
    """"mpc"/"lqr" must be listed for CLI choices even though constructing
    either is expensive and nothing here has imported
    lunar_lander_lab.controllers.mpc or .lqr."""
    assert set(controller_names()) >= {
        "heuristic", "scheduled", "rl", "mpc", "lqr", "sac", "dqn", "td3",
    }


def test_heuristic_builds_the_right_class():
    assert isinstance(build_controller("heuristic"), HeuristicController)


def test_scheduled_builds_the_right_class():
    assert isinstance(build_controller("scheduled"), ScheduledHeuristicController)


def test_heuristic_gains_override_rejects_unknown_gain(tmp_path):
    bad_gains = tmp_path / "gains.json"
    bad_gains.write_text('{"NOT_A_REAL_GAIN": 1.0}')
    with pytest.raises(ValueError, match="unknown gain"):
        build_controller("heuristic", gains_path=str(bad_gains))


def test_unknown_controller_raises():
    with pytest.raises(ValueError, match="Unknown controller: bogus"):
        build_controller("bogus")


def test_rl_builder_loads_an_explicit_model_name(tmp_path, monkeypatch):
    """build_controller("rl", model_name=...) should reach RLAgent.load with
    that name, same as cli.py's old "rl" branch did."""
    resolved = {}
    monkeypatch.setattr(
        "lunar_lander_lab.controllers.rl_agent.PPO.load",
        lambda path: resolved.setdefault("path", str(path)) and None or "model",
    )
    checkpoint = tmp_path / "some_model.zip"
    checkpoint.write_text("stub")

    build_controller("rl", model_name=str(checkpoint))

    assert resolved["path"] == str(checkpoint)


@pytest.mark.parametrize(
    "controller_name,sb3_class",
    [("sac", "SAC"), ("dqn", "DQN"), ("td3", "TD3")],
)
def test_rl_variant_builders_load_via_their_own_algo_class(
    controller_name, sb3_class, tmp_path, monkeypatch
):
    """Each non-PPO registry entry must load through its own SB3 class, not
    silently fall back to PPO.load -- otherwise "sac"/"dqn"/"td3" would all
    resolve to the same (wrong) checkpoint format."""
    resolved = {}
    monkeypatch.setattr(
        f"lunar_lander_lab.controllers.rl_agent.{sb3_class}.load",
        lambda path: resolved.setdefault("path", str(path)) and None or "model",
    )
    checkpoint = tmp_path / "some_model.zip"
    checkpoint.write_text("stub")

    controller = build_controller(controller_name, model_name=str(checkpoint))

    assert resolved["path"] == str(checkpoint)
    assert controller.algo.__name__ == sb3_class


@pytest.mark.slow
def test_mpc_builds_lazily_via_the_registry():
    """Constructing MPCController measures constants off a live env, so this
    is the one registry test that touches Gymnasium and stays behind -m slow."""
    from lunar_lander_lab.controllers.mpc import MPCController

    controller = build_controller("mpc")
    assert isinstance(controller, MPCController)
    assert "mpc" in controller_names()


@pytest.mark.slow
def test_lqr_builds_lazily_via_the_registry():
    """Same lazy-import contract as "mpc" -- LQRController also measures
    live-env constants by default."""
    from lunar_lander_lab.controllers.lqr import LQRController

    controller = build_controller("lqr")
    assert isinstance(controller, LQRController)
    assert "lqr" in controller_names()
