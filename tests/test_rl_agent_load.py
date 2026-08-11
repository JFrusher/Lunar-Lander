"""RLAgent.load()'s path resolution, without training or loading a real model."""

import pytest

from lunar_lander_lab.controllers.rl_agent import DEFAULT_HYPERPARAMS, RLAgent


@pytest.fixture
def fake_ppo_load(monkeypatch):
    """Capture the path load() resolves to, instead of parsing a real .zip."""
    resolved = {}
    monkeypatch.setattr(
        "lunar_lander_lab.controllers.rl_agent.PPO.load",
        lambda path: resolved.setdefault("path", str(path)) and None or "model",
    )
    return resolved


def test_existing_file_path_is_used_as_is(tmp_path, fake_ppo_load):
    checkpoint = tmp_path / "some_model.zip"
    checkpoint.write_text("stub")

    RLAgent().load(str(checkpoint))
    assert fake_ppo_load["path"] == str(checkpoint)


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    """Point the fallback at a temp runs/ dir.

    `latest_run_file` binds RUNS_DIR as a default argument at import time, so
    patching the module attribute would not reach it — redirect at the call
    site in rl_agent instead, which still exercises the real resolution.
    """
    from lunar_lander_lab.utils.paths import latest_run_file

    monkeypatch.setattr(
        "lunar_lander_lab.controllers.rl_agent.latest_run_file",
        lambda kind, filename: latest_run_file(kind, filename, base=tmp_path),
    )
    return tmp_path


def test_bare_name_falls_back_to_the_latest_run(runs_dir, fake_ppo_load):
    """A name like "ppo_lunar_lander" resolves against runs/train/*/<name>.zip."""
    older = runs_dir / "train" / "20260101_000000"
    newer = runs_dir / "train" / "20260609_120000"
    for d in (older, newer):
        d.mkdir(parents=True)
        (d / "ppo_lunar_lander.zip").write_text("stub")

    RLAgent().load("ppo_lunar_lander")

    assert fake_ppo_load["path"] == str(newer / "ppo_lunar_lander.zip")


def test_unknown_name_raises_file_not_found(runs_dir, fake_ppo_load):
    with pytest.raises(FileNotFoundError):
        RLAgent().load("no_such_checkpoint")


def test_get_action_without_a_model_is_a_clear_error():
    with pytest.raises(RuntimeError, match="No model loaded"):
        RLAgent().get_action([0.0] * 8)


def test_tensorboard_log_is_off_unless_asked_for():
    """Sweeps train dozens of models per run and shouldn't all write curves,
    so the default must stay clean -- and must not leak into the shared
    DEFAULT_HYPERPARAMS dict, which every other caller reads."""
    assert "tensorboard_log" not in DEFAULT_HYPERPARAMS


def test_tensorboard_log_reaches_ppo_when_requested(tmp_path, monkeypatch):
    captured = {}

    class _StubPPO:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def learn(self, **_):
            pass

        def save(self, _path):
            pass

    monkeypatch.setattr("lunar_lander_lab.controllers.rl_agent.PPO", _StubPPO)
    monkeypatch.setattr(
        "lunar_lander_lab.controllers.rl_agent.new_run_dir", lambda _kind: tmp_path
    )

    RLAgent().train(total_timesteps=1, tensorboard_log=str(tmp_path / "tb"))

    assert captured["tensorboard_log"] == str(tmp_path / "tb")
    # The shared default dict must be untouched by that call.
    assert "tensorboard_log" not in DEFAULT_HYPERPARAMS
