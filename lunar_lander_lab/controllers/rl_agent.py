"""PPO-based reinforcement learning controller (Stable-Baselines3)."""

from pathlib import Path
from typing import Callable, Optional, Sequence

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from .base import BaseController
from ..utils.paths import latest_run_file, new_run_dir

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


class RLAgent(BaseController):
    """Wraps a Stable-Baselines3 PPO model behind the BaseController interface."""

    def __init__(self, model_path: Optional[str] = None):
        self.model: Optional[PPO] = None
        if model_path is not None:
            self.load(model_path)

    def train(
        self,
        env_name: str = "LunarLander-v3",
        total_timesteps: int = 100_000,
        save_path: str = "ppo_lunar_lander",
        hyperparams: Optional[dict] = None,
        env_wrapper: Optional[Callable[[gym.Env], gym.Env]] = None,
    ) -> str:
        """Train a PPO model from scratch and save the weights to runs/train/<timestamp>/."""
        params = {**DEFAULT_HYPERPARAMS, **(hyperparams or {})}

        def make_env():
            env = gym.make(env_name)
            return env_wrapper(env) if env_wrapper else env

        env = DummyVecEnv([make_env])

        self.model = PPO(env=env, **params)
        self.model.learn(total_timesteps=total_timesteps)

        run_dir = new_run_dir("train")
        full_path = run_dir / f"{save_path}.zip"
        self.model.save(str(full_path))
        env.close()
        return str(full_path)

    def load(self, model_path: str) -> None:
        """Load a previously trained PPO model.

        `model_path` may be an existing file path, or a checkpoint name
        (e.g. "ppo_lunar_lander") to resolve against the most recent
        runs/train/*/<name>.zip.
        """
        path = Path(model_path)
        if not path.exists():
            path = latest_run_file("train", f"{model_path}.zip")
        self.model = PPO.load(str(path))

    def get_action(self, observation: Sequence[float]) -> int:
        if self.model is None:
            raise RuntimeError("No model loaded. Call train() or load() first.")
        action, _ = self.model.predict(observation, deterministic=True)
        return int(action)
