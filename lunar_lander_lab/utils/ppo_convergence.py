"""PPO training-timestep convergence check.

Measures whether `RLAgent.train()` has actually converged at a given
`total_timesteps` budget, instead of assuming a round number is enough.
Trains at each checkpoint in `TIMESTEP_CHECKPOINTS` with a fixed seed
(isolating timesteps as the only variable) and evaluates with the same
natural-reward evaluator the time-penalty sweep uses.
"""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from ..controllers.rl_agent import RLAgent
from .paths import new_run_dir
from .time_penalty import evaluate_controller_natural

TIMESTEP_CHECKPOINTS: List[int] = [100_000, 200_000, 400_000, 700_000, 1_000_000]

# Same palette convention as time_penalty.py's trade-off plot.
_SURFACE = "#fcfcfb"
_GRID = "#e1e0d9"
_INK_PRIMARY = "#0b0b0b"
_INK_MUTED = "#898781"
_LINE_COLOR = "#2a78d6"


def recommend_timesteps(
    df: pd.DataFrame,
    success_tolerance_pct: float = 5.0,
    reward_fraction: float = 0.95,
) -> int:
    """Pick the smallest `total_timesteps` that's already near the run's peak.

    "Near peak" means: within `success_tolerance_pct` points of the best
    observed success rate, and at least `reward_fraction` of the best
    observed mean reward. Avoids recommending a needlessly large budget
    when a smaller one already reaches the same plateau.
    """
    best_success = df["success_rate_pct"].max()
    best_reward = df["mean_reward"].max()

    near_peak = df[
        (df["success_rate_pct"] >= best_success - success_tolerance_pct)
        & (df["mean_reward"] >= reward_fraction * best_reward)
    ]
    return int(near_peak["total_timesteps"].min())


def _plot_convergence(df: pd.DataFrame, path: Path) -> None:
    fig, (ax_reward, ax_success) = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=_SURFACE)

    for ax in (ax_reward, ax_success):
        ax.set_facecolor(_SURFACE)
        ax.grid(True, color=_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color(_GRID)
        ax.tick_params(colors=_INK_MUTED)

    df = df.sort_values("total_timesteps")
    ax_reward.plot(
        df["total_timesteps"], df["mean_reward"],
        marker="o", markersize=8, linewidth=2, color=_LINE_COLOR,
    )
    ax_reward.set_xlabel("training timesteps", color=_INK_PRIMARY)
    ax_reward.set_ylabel("mean reward (natural)", color=_INK_PRIMARY)
    ax_reward.set_title("Reward vs. Training Timesteps", color=_INK_PRIMARY)

    ax_success.plot(
        df["total_timesteps"], df["success_rate_pct"],
        marker="o", markersize=8, linewidth=2, color=_LINE_COLOR,
    )
    ax_success.set_xlabel("training timesteps", color=_INK_PRIMARY)
    ax_success.set_ylabel("success rate (%)", color=_INK_PRIMARY)
    ax_success.set_title("Success Rate vs. Training Timesteps", color=_INK_PRIMARY)

    fig.suptitle("PPO Convergence Check", color=_INK_PRIMARY, y=1.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=_SURFACE, bbox_inches="tight")
    plt.close(fig)


def run_ppo_convergence_check(
    timestep_checkpoints: List[int] = TIMESTEP_CHECKPOINTS,
    eval_episodes: int = 30,
    seed: int = 0,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Train PPO from scratch at each of `timestep_checkpoints`, fixed seed,
    and evaluate natural reward/success rate at each budget."""
    run_dir = Path(output_dir) if output_dir else new_run_dir("ppo_convergence")
    rows = []

    for total_timesteps in timestep_checkpoints:
        print(f"\n=== PPO, total_timesteps={total_timesteps} ===")
        agent = RLAgent()
        saved_path = agent.train(
            total_timesteps=total_timesteps,
            save_path=f"ppo_convergence_{total_timesteps}",
            hyperparams={"seed": seed},
        )
        metrics = evaluate_controller_natural(agent, eval_episodes, seed_start=seed)
        rows.append({"total_timesteps": total_timesteps, "detail": saved_path, **metrics})

    df_results = pd.DataFrame(rows)
    csv_path = run_dir / "ppo_convergence_results.csv"
    df_results.to_csv(csv_path, index=False)

    plot_path = run_dir / "ppo_convergence.png"
    _plot_convergence(df_results, plot_path)

    recommended = recommend_timesteps(df_results)

    print(f"\nResults:    {csv_path}")
    print(f"Plot:       {plot_path}")
    print(f"Recommended total_timesteps: {recommended}")
    print(df_results.to_string(index=False))
    return df_results


if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "total_timesteps": [100_000, 200_000, 400_000],
            "mean_reward": [150.0, 240.0, 245.0],
            "success_rate_pct": [40.0, 95.0, 97.0],
        }
    )
    assert recommend_timesteps(df) == 200_000, recommend_timesteps(df)
    print("ppo_convergence self-check OK")
