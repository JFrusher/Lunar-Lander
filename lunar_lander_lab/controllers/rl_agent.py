"""Reinforcement-learning controllers (Stable-Baselines3): PPO, SAC, DQN, TD3."""

from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Type, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import DQN, PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv

from ..utils.evaluation import CONTINUOUS_ENV_KWARGS
from ..utils.paths import latest_run_file, new_run_dir
from .base import BaseController
from .registry import register_controller

# name -> algo class. Single source of truth for the CLI's --algo flag and
# for this module's own registry builders. "ppo" keeps registering under the
# controller name "rl" for backward compatibility (every doc/test predates
# this map); sac/dqn/td3 register under their algo name directly.
ALGOS: Dict[str, Type[BaseAlgorithm]] = {"ppo": PPO, "sac": SAC, "dqn": DQN, "td3": TD3}
DEFAULT_MODEL_NAMES: Dict[str, str] = {name: f"{name}_lunar_lander" for name in ALGOS}
DEFAULT_MODEL_NAME = DEFAULT_MODEL_NAMES["ppo"]

# SAC/TD3 are off-policy and continuous-only; DQN is off-policy and
# discrete-only. PPO (on-policy) accepts either, matching every result
# recorded before this module supported more than PPO.
_CONTINUOUS_ONLY = (SAC, TD3)
_DISCRETE_ONLY = (DQN,)

# Default PPO hyperparameters for LunarLander. Challenged by a 30-config
# Latin-Hypercube sweep at 1M timesteps (see `ppo_search.py`); the winner was
# retrained at 3 seeds and measured on held-out episodes against these:
#
#            reward (3 seeds)   success   landing steps
#   default    261.2 ± 4.8       99.3%        321
#   swept      261.3 ± 15.8      89.3%        282
#
# Indistinguishable on reward; the default is better on success rate and
# three times more seed-stable, which is why it stays. The swept config is
# ~12% faster and would be the better pick if landing speed were the
# objective. Both single-seed figures that earlier looked decisive here
# (277.0 for the default, 269.4 for the swept config) were lucky draws.
DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "n_steps": 1024,
    "batch_size": 64,
    "n_epochs": 4,
    "gamma": 0.999,
    "gae_lambda": 0.98,
    "ent_coef": 0.01,
    "verbose": 1,
}

# SAC/DQN/TD3 have never been tuned here (unlike PPO above), so their
# defaults are just SB3's own -- the minimal kwargs every SB3 algorithm
# accepts, rather than invented numbers dressed up as validated ones.
_MINIMAL_HYPERPARAMS = {"policy": "MlpPolicy", "verbose": 1}
ALGO_DEFAULT_HYPERPARAMS: Dict[Type[BaseAlgorithm], dict] = {
    PPO: DEFAULT_HYPERPARAMS,
    SAC: _MINIMAL_HYPERPARAMS,
    DQN: _MINIMAL_HYPERPARAMS,
    TD3: _MINIMAL_HYPERPARAMS,
}


def _default_env_kwargs(algo: Type[BaseAlgorithm]) -> dict:
    return dict(CONTINUOUS_ENV_KWARGS) if algo in _CONTINUOUS_ONLY else {}


def _check_env_compatible(algo: Type[BaseAlgorithm], env_kwargs: dict) -> None:
    """Raise a clear error for an algo/action-space mismatch instead of
    SB3's own (an assertion buried inside the algorithm's constructor)."""
    is_continuous = bool(env_kwargs.get("continuous"))
    if algo in _CONTINUOUS_ONLY and not is_continuous:
        raise ValueError(
            f"{algo.__name__} needs a continuous action space; pass "
            f"env_kwargs={{'continuous': True}} (or omit env_kwargs to get it by default)"
        )
    if algo in _DISCRETE_ONLY and is_continuous:
        raise ValueError(f"{algo.__name__} needs a discrete action space, not continuous")


class RLAgent(BaseController):
    """Wraps a Stable-Baselines3 model behind the BaseController interface.

    `algo` selects which SB3 algorithm this agent trains/loads -- PPO by
    default, matching every result recorded before this took an `algo`
    argument. PPO accepts either action space; SAC/TD3 require continuous,
    DQN requires discrete (`train()` defaults the env to whichever the
    algorithm needs, and rejects an explicit `env_kwargs` that contradicts
    it).
    """

    def __init__(self, model_path: Optional[str] = None, algo: Optional[Type[BaseAlgorithm]] = None):
        # `algo or PPO` (not a `= PPO` default) so it's resolved from this
        # module's globals at call time, not bound once at import time --
        # tests monkeypatch `rl_agent.PPO` itself to stub out real training.
        self.algo = algo or PPO
        self.model: Optional[BaseAlgorithm] = None
        if model_path is not None:
            self.load(model_path)

    def train(
        self,
        env_name: str = "LunarLander-v3",
        total_timesteps: int = 100_000,
        save_path: str = "ppo_lunar_lander",
        hyperparams: Optional[dict] = None,
        env_wrapper: Optional[Callable[[gym.Env], gym.Env]] = None,
        tensorboard_log: Optional[str] = None,
        env_kwargs: Optional[dict] = None,
    ) -> str:
        """Train a model from scratch and save the weights to runs/train/<timestamp>/.

        `env_kwargs` defaults to whatever action space `self.algo` needs
        (continuous for SAC/TD3, discrete for PPO/DQN) -- pass it explicitly
        only to override that, e.g. continuous-PPO experiments like
        `continuous_compare.py`.

        Pass `tensorboard_log` (a directory) to record learning curves. Off
        by default: the sweeps in `utils/` train dozens of models per run and
        don't all want the extra writes. Worth switching on whenever a
        *single* run's behaviour over time matters -- every collapse this
        project has diagnosed so far was read off a final number after the
        fact, when a curve would have shown it happening.
        """
        env_kwargs = _default_env_kwargs(self.algo) if env_kwargs is None else env_kwargs
        _check_env_compatible(self.algo, env_kwargs)

        params = {**ALGO_DEFAULT_HYPERPARAMS.get(self.algo, _MINIMAL_HYPERPARAMS), **(hyperparams or {})}
        if tensorboard_log is not None:
            params["tensorboard_log"] = tensorboard_log

        def make_env():
            env = gym.make(env_name, **env_kwargs)
            return env_wrapper(env) if env_wrapper else env

        env = DummyVecEnv([make_env])

        self.model = self.algo(env=env, **params)
        self.model.learn(total_timesteps=total_timesteps)

        run_dir = new_run_dir("train")
        full_path = run_dir / f"{save_path}.zip"
        self.model.save(str(full_path))
        env.close()
        return str(full_path)

    def load(self, model_path: str) -> None:
        """Load a previously trained model of this agent's `algo`.

        `model_path` may be an existing file path, or a checkpoint name
        (e.g. "ppo_lunar_lander") to resolve against the most recent
        runs/train/*/<name>.zip.
        """
        path = Path(model_path)
        if not path.exists():
            path = latest_run_file("train", f"{model_path}.zip")
        self.model = self.algo.load(str(path))

    def get_action(self, observation: Sequence[float]) -> Union[int, np.ndarray]:
        """Return an action of whatever type this model's action space uses.

        Discrete gives a plain int (what every controller in this repo has
        returned until now); a Box action space gives the raw 2-vector. The
        model's own `action_space` decides, so a continuous checkpoint can be
        loaded and evaluated through the same code path as a discrete one --
        `int(action)` would silently truncate the vector to its first element.
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call train() or load() first.")
        action, _ = self.model.predict(observation, deterministic=True)
        if isinstance(self.model.action_space, spaces.Discrete):
            return int(action)
        return action

    @property
    def env_kwargs(self) -> dict:
        """The env kwargs this agent's *loaded* model actually needs.

        Unlike `_default_env_kwargs` (used by `train()` before a model
        exists), this reads the real `action_space` off `self.model` once
        loaded, so it is correct even for a checkpoint trained with an
        explicit override (e.g. continuous-PPO).
        """
        if self.model is not None and not isinstance(self.model.action_space, spaces.Discrete):
            return dict(CONTINUOUS_ENV_KWARGS)
        return {}


@register_controller("rl")
def _build_ppo(model_name=None, gains_path=None) -> RLAgent:
    agent = RLAgent(algo=PPO)
    agent.load(model_name or DEFAULT_MODEL_NAMES["ppo"])
    return agent


@register_controller("sac")
def _build_sac(model_name=None, gains_path=None) -> RLAgent:
    agent = RLAgent(algo=SAC)
    agent.load(model_name or DEFAULT_MODEL_NAMES["sac"])
    return agent


@register_controller("dqn")
def _build_dqn(model_name=None, gains_path=None) -> RLAgent:
    agent = RLAgent(algo=DQN)
    agent.load(model_name or DEFAULT_MODEL_NAMES["dqn"])
    return agent


@register_controller("td3")
def _build_td3(model_name=None, gains_path=None) -> RLAgent:
    agent = RLAgent(algo=TD3)
    agent.load(model_name or DEFAULT_MODEL_NAMES["td3"])
    return agent
