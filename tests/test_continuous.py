"""Phase 8b of tmp/SPEED_ROADMAP.md: does throttle control beat bang-bang?

Continuous-PPO changes ONE thing versus every other RL result in this repo --
the action space. Same algorithm, same 1M-timestep budget, same evaluation.
"""

import gymnasium as gym
import numpy as np
import pytest

from lunar_lander_lab.controllers.heuristic import HeuristicController
from lunar_lander_lab.utils.evaluation import (
    CONTINUOUS_ENV_KWARGS,
    count_engine_frames,
    evaluate_controller_natural,
)


def test_continuous_env_kwargs_select_the_box_action_space():
    env = gym.make("LunarLander-v3", **CONTINUOUS_ENV_KWARGS)
    try:
        assert isinstance(env.action_space, gym.spaces.Box)
        assert env.action_space.shape == (2,)
        # Observation space is unchanged -- that's the point of the comparison.
        assert env.observation_space.shape == (8,)
    finally:
        env.close()


class _DiscreteActions:
    def __init__(self, actions):
        self.actions, self.index = list(actions), 0

    def get_action(self, _obs):
        a = self.actions[self.index] if self.index < len(self.actions) else 0
        self.index += 1
        return a


@pytest.mark.parametrize(
    "action, main, side",
    [
        (0, 0, 0),
        (2, 1, 0),
        (1, 0, 1),
        (3, 0, 1),
    ],
)
def test_count_engine_frames_discrete(action, main, side):
    assert count_engine_frames(action) == (main, side)


@pytest.mark.parametrize(
    "action, main, side",
    [
        # LunarLander fires the main engine only for action[0] > 0, and a side
        # engine only for |action[1]| > 0.5 -- fuel accounting has to use the
        # env's own thresholds or it counts frames that never burned.
        ([0.0, 0.0], 0, 0),
        ([-1.0, 0.0], 0, 0),
        ([0.5, 0.0], 1, 0),
        ([1.0, 0.6], 1, 1),
        ([-1.0, -0.9], 0, 1),
        ([0.0, 0.5], 0, 0),
    ],
)
def test_count_engine_frames_continuous(action, main, side):
    assert count_engine_frames(np.array(action, dtype=np.float32)) == (main, side)


def test_evaluate_accepts_env_kwargs_and_still_scores_a_discrete_controller():
    """env_kwargs is additive: omitting it must behave exactly as before."""
    plain = evaluate_controller_natural(HeuristicController(), 2, seed_start=50_000)
    explicit = evaluate_controller_natural(
        HeuristicController(), 2, seed_start=50_000, env_kwargs={}
    )
    assert plain["mean_reward"] == pytest.approx(explicit["mean_reward"])


def test_rl_agent_returns_a_vector_for_a_box_action_space(monkeypatch):
    """RLAgent hardcoded int(action), which silently truncates a 2-vector.
    The action space the loaded model was trained on decides the type."""
    from lunar_lander_lab.controllers.rl_agent import RLAgent

    class _StubModel:
        action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

        def predict(self, _obs, deterministic=True):
            return np.array([0.4, -0.7], dtype=np.float32), None

    agent = RLAgent()
    agent.model = _StubModel()
    action = agent.get_action([0.0] * 8)

    assert isinstance(action, np.ndarray)
    assert action.shape == (2,)
    assert agent.model.action_space.contains(action)


def test_rl_agent_still_returns_int_for_a_discrete_action_space():
    from lunar_lander_lab.controllers.rl_agent import RLAgent

    class _StubModel:
        action_space = gym.spaces.Discrete(4)

        def predict(self, _obs, deterministic=True):
            return np.int64(2), None

    agent = RLAgent()
    agent.model = _StubModel()

    assert agent.get_action([0.0] * 8) == 2
    assert isinstance(agent.get_action([0.0] * 8), int)


@pytest.mark.slow
def test_evaluate_runs_a_continuous_controller_end_to_end():
    """A constant full-throttle policy should burn main-engine frames on every
    step -- proves the continuous path actually reaches the fuel counters."""

    class _FullThrottle:
        def get_action(self, _obs):
            return np.array([1.0, 0.0], dtype=np.float32)

    m = evaluate_controller_natural(
        _FullThrottle(), 1, seed_start=50_000, env_kwargs=CONTINUOUS_ENV_KWARGS
    )
    assert m["main_engine_frames"] > 0
    assert m["side_engine_frames"] == 0
