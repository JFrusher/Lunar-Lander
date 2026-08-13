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


# --- Arc 4 Phase B: algo parametrization (SAC/DQN/TD3 alongside PPO) -------


def test_default_algo_is_still_ppo():
    """`RLAgent()` with no `algo` must behave exactly as before this arg
    existed -- every earlier result in this repo assumed that."""
    from stable_baselines3 import PPO

    assert RLAgent().algo is PPO


def test_default_env_kwargs_by_algo():
    from stable_baselines3 import DQN, PPO, SAC, TD3

    from lunar_lander_lab.controllers.rl_agent import _default_env_kwargs

    assert _default_env_kwargs(PPO) == {}
    assert _default_env_kwargs(DQN) == {}
    assert _default_env_kwargs(SAC) == {"continuous": True}
    assert _default_env_kwargs(TD3) == {"continuous": True}


def test_dqn_rejects_an_explicit_continuous_env_override():
    from stable_baselines3 import DQN

    with pytest.raises(ValueError, match="discrete"):
        RLAgent(algo=DQN).train(total_timesteps=1, env_kwargs={"continuous": True})


def test_sac_rejects_an_explicit_discrete_env_override():
    from stable_baselines3 import SAC

    with pytest.raises(ValueError, match="continuous"):
        RLAgent(algo=SAC).train(total_timesteps=1, env_kwargs={})


def test_env_kwargs_defaults_to_discrete_with_no_model():
    assert RLAgent().env_kwargs == {}


def test_env_kwargs_reflects_a_loaded_continuous_model():
    from gymnasium import spaces

    class _FakeModel:
        action_space = spaces.Box(-1, 1, (2,))

    agent = RLAgent()
    agent.model = _FakeModel()
    assert agent.env_kwargs == {"continuous": True}


def test_env_kwargs_reflects_a_loaded_discrete_model():
    from gymnasium import spaces

    class _FakeModel:
        action_space = spaces.Discrete(4)

    agent = RLAgent()
    agent.model = _FakeModel()
    assert agent.env_kwargs == {}


@pytest.mark.slow
def test_train_load_and_act_round_trips_for_a_non_ppo_algorithm(tmp_path, monkeypatch):
    """DQN end-to-end: train() -> save -> load() -> get_action(). Proves the
    algo parametrization actually reaches a second SB3 algorithm family
    through the real SB3 API, not just PPO with an unused constructor arg."""
    from stable_baselines3 import DQN

    monkeypatch.setattr(
        "lunar_lander_lab.controllers.rl_agent.new_run_dir", lambda _kind: tmp_path
    )

    agent = RLAgent(algo=DQN)
    saved_path = agent.train(
        total_timesteps=200,
        save_path="dqn_smoke",
        hyperparams={"buffer_size": 1_000, "learning_starts": 10},
    )

    loaded = RLAgent(algo=DQN)
    loaded.load(saved_path)
    assert loaded.get_action([0.0] * 8) in {0, 1, 2, 3}
