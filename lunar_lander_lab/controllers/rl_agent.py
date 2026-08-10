"""PPO-based reinforcement learning controller (Stable-Baselines3)."""

from pathlib import Path
from typing import Optional, Sequence

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from .base import BaseController

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Default PPO hyperparameters, tuned as a reasonable starting point for LunarLander.
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
    ) -> str:
        """Train a PPO model from scratch and save the weights to models/."""
        params = {**DEFAULT_HYPERPARAMS, **(hyperparams or {})}
        env = DummyVecEnv([lambda: gym.make(env_name)])

        self.model = PPO(env=env, **params)
        self.model.learn(total_timesteps=total_timesteps)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        full_path = MODELS_DIR / save_path
        self.model.save(str(full_path))
        env.close()
        return str(full_path.with_suffix(".zip"))

    def load(self, model_path: str) -> None:
        """Load a previously trained PPO model."""
        path = Path(model_path)
        if not path.is_absolute() and not path.exists():
            path = MODELS_DIR / model_path
        self.model = PPO.load(str(path))

    def get_action(self, observation: Sequence[float]) -> int:
        if self.model is None:
            raise RuntimeError("No model loaded. Call train() or load() first.")
        action, _ = self.model.predict(observation, deterministic=True)
        return int(action)
