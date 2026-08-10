"""Time-penalty reward shaping + sweep across controller types."""

import math
from pathlib import Path
from typing import Dict, List, Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd

from ..controllers.base import BaseController
from ..controllers.rl_agent import RLAgent
from .paths import new_run_dir
from .pid_search import run_monte_carlo

SUCCESS_REWARD_THRESHOLD = 200

TIME_PENALTY_COEFS: List[float] = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]

# Fixed-order categorical pair (dataviz skill default palette, slots 1-2:
# blue/orange), validated CVD-safe adjacent pair in both light and dark modes.
_CONTROLLER_COLORS = {"Heuristic": "#2a78d6", "RL (PPO)": "#eb6834"}
_SURFACE = "#fcfcfb"
_GRID = "#e1e0d9"
_INK_PRIMARY = "#0b0b0b"
_INK_MUTED = "#898781"


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


def _plot_tradeoff(df: pd.DataFrame, path: Path) -> None:
    fig, (ax_steps, ax_reward) = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=_SURFACE)

    for ax in (ax_steps, ax_reward):
        ax.set_facecolor(_SURFACE)
        ax.grid(True, color=_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color(_GRID)
        ax.tick_params(colors=_INK_MUTED)

    for controller, group in df.groupby("controller"):
        group = group.sort_values("penalty")
        color = _CONTROLLER_COLORS.get(controller, "#4a3aa7")
        ax_steps.plot(
            group["penalty"], group["avg_landing_steps"],
            marker="o", markersize=8, linewidth=2, color=color, label=controller,
        )
        ax_reward.plot(
            group["penalty"], group["mean_reward"],
            marker="o", markersize=8, linewidth=2, color=color, label=controller,
        )

    ax_steps.set_xlabel("time penalty (per step)", color=_INK_PRIMARY)
    ax_steps.set_ylabel("avg landing steps (successful episodes)", color=_INK_PRIMARY)
    ax_steps.set_title("Landing Speed vs. Penalty", color=_INK_PRIMARY)

    ax_reward.set_xlabel("time penalty (per step)", color=_INK_PRIMARY)
    ax_reward.set_ylabel("mean reward (natural)", color=_INK_PRIMARY)
    ax_reward.set_title("Reward vs. Penalty", color=_INK_PRIMARY)

    handles, labels = ax_steps.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06), frameon=False)

    fig.suptitle("Time-Penalty Sweep: Landing Speed / Reward Trade-off", color=_INK_PRIMARY, y=1.16)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=_SURFACE, bbox_inches="tight")
    plt.close(fig)


def run_time_penalty_sweep(
    penalties: List[float] = TIME_PENALTY_COEFS,
    rl_timesteps: int = 150_000,
    pid_samples: int = 200,
    pid_episodes: int = 30,
    eval_episodes: int = 30,
    seed: int = 0,
    n_jobs: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Sweep `penalties`, training/tuning both controllers at each level, and
    report natural-reward metrics + a trade-off plot."""
    sweep_dir = Path(output_dir) if output_dir else new_run_dir("time_penalty_sweep")
    rows = []

    for p in penalties:
        print(f"\n=== Heuristic, penalty={p} ===")
        gains_df = run_monte_carlo(
            n_samples=pid_samples,
            episodes_per_set=pid_episodes,
            seed=seed,
            time_penalty=p,
            output_dir=str(sweep_dir / f"heuristic_penalty_{p}"),
            n_jobs=n_jobs,
        )
        penalized_score = gains_df["mean_reward"] - p * gains_df["avg_steps"]
        best = gains_df.loc[penalized_score.idxmax()]
        rows.append(
            {
                "controller": "Heuristic",
                "penalty": p,
                "mean_reward": float(best["mean_reward"]),
                "success_rate_pct": float(best["success_rate_pct"]),
                "crash_rate_pct": float(best["crash_rate_pct"]),
                "avg_landing_steps": float(best["avg_steps_success"]),
                "detail": str(sweep_dir / f"heuristic_penalty_{p}" / "best_gains.json"),
            }
        )

    for p in penalties:
        print(f"\n=== RL (PPO), penalty={p} ===")
        agent = RLAgent()
        saved_path = agent.train(
            total_timesteps=rl_timesteps,
            save_path=f"ppo_penalty_{p}",
            env_wrapper=lambda env, p=p: TimePenaltyWrapper(env, p),
        )
        metrics = evaluate_controller_natural(agent, eval_episodes, seed_start=seed)
        rows.append({"controller": "RL (PPO)", "penalty": p, "detail": saved_path, **metrics})

    df_results = pd.DataFrame(rows)
    csv_path = sweep_dir / "time_penalty_sweep_results.csv"
    df_results.to_csv(csv_path, index=False)

    plot_path = sweep_dir / "time_penalty_tradeoff.png"
    _plot_tradeoff(df_results, plot_path)

    print(f"\nResults: {csv_path}")
    print(f"Plot:    {plot_path}")
    print(df_results.to_string(index=False))
    return df_results


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
