"""Time-penalty reward shaping + sweep across controller types."""

import math
from typing import Dict, List

import gymnasium as gym

from ..controllers.base import BaseController

SUCCESS_REWARD_THRESHOLD = 200

TIME_PENALTY_COEFS: List[float] = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]


class TimePenaltyWrapper(gym.Wrapper):
    """Subtracts a fixed penalty from the reward every timestep.

    Used only when building the RL *training* env — never for evaluation.
    """

    def __init__(self, env: gym.Env, penalty_per_step: float):
        super().__init__(env)
        self.penalty_per_step = penalty_per_step

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward - self.penalty_per_step, terminated, truncated, info


def evaluate_controller_natural(
    controller: BaseController,
    num_episodes: int,
    env_name: str = "LunarLander-v3",
    seed_start: int = 0,
) -> Dict[str, float]:
    """Evaluate `controller` over `num_episodes` seeded episodes with natural (unpenalized) reward."""
    env = gym.make(env_name)
    successes = crashes = 0
    rewards = []
    success_steps = []

    for seed in range(seed_start, seed_start + num_episodes):
        observation, _ = env.reset(seed=seed)
        total_reward = 0.0
        steps = 0
        terminated = truncated = False

        while not (terminated or truncated):
            action = controller.get_action(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            steps += 1

        rewards.append(total_reward)
        if total_reward >= SUCCESS_REWARD_THRESHOLD:
            successes += 1
            success_steps.append(steps)
        elif terminated and total_reward <= -100:
            crashes += 1

    env.close()

    return {
        "mean_reward": sum(rewards) / len(rewards),
        "success_rate_pct": 100 * successes / num_episodes,
        "crash_rate_pct": 100 * crashes / num_episodes,
        "avg_landing_steps": (
            sum(success_steps) / len(success_steps) if success_steps else math.nan
        ),
    }


if __name__ == "__main__":
    from ..controllers.heuristic import HeuristicController

    env_a = gym.make("LunarLander-v3")
    env_b = TimePenaltyWrapper(gym.make("LunarLander-v3"), penalty_per_step=0.05)
    env_a.reset(seed=0)
    env_b.reset(seed=0)
    _, reward_a, *_ = env_a.step(0)
    _, reward_b, *_ = env_b.step(0)
    env_a.close()
    env_b.close()
    assert abs((reward_a - 0.05) - reward_b) < 1e-9, (reward_a, reward_b)

    metrics = evaluate_controller_natural(HeuristicController(), num_episodes=3)
    assert set(metrics) == {"mean_reward", "success_rate_pct", "crash_rate_pct", "avg_landing_steps"}
    assert not math.isnan(metrics["avg_landing_steps"]), "heuristic should land at least once in 3 tries"

    print("time_penalty self-check OK:", metrics)
