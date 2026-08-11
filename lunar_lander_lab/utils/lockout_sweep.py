"""Engine-lockout gating sweep.

Does forcing free-fall-plus-attitude-only control for the early part of
descent -- instead of letting the controller hover-correct the whole way
down -- produce a genuinely faster landing? Phase 2 of
tmp/SPEED_ROADMAP.md.

Two alternative gate forms are compared, not combined: `lockout_steps`
(block the main engine for the first N steps of every episode) and
`altitude_threshold` (block it while still above a given height). Which one
is actually better -- or whether either beats doing nothing -- is the open
question the roadmap named explicitly.
"""

import multiprocessing
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from ..controllers.heuristic import HeuristicController
from .paths import new_run_dir
from .pid_search import HOLDOUT_SEED_START
from .time_penalty import (
    _CONTROLLER_COLORS,
    _INK_MUTED,
    _INK_PRIMARY,
    _SURFACE,
    EngineLockoutWrapper,
    _style_axes,
    evaluate_controller_natural,
)

# The first pass ran [20, 50, 100]: 20 won, 50 already broke. So "20 is best"
# was best-of-one-viable-value, not a located optimum -- the whole interesting
# region is below 50 and had a single sample in it. 5/10/15/30/40 added on the
# second pass to actually resolve the curve between "free win" and "collapse".
STEP_THRESHOLDS: List[int] = [5, 10, 15, 20, 30, 40, 50, 100]

# All three failed outright on both legs in the first pass (0% success for
# every altitude gate on RL; 69% and below on the heuristic). Kept as the
# record of what was tested -- pass `altitude_thresholds=[]` to skip retraining
# them.
ALTITUDE_THRESHOLDS: List[float] = [1.0, 0.6, 0.3]


def lockout_grid(
    step_thresholds: Optional[List[int]] = None,
    altitude_thresholds: Optional[List[float]] = None,
) -> List[Dict[str, float]]:
    """The gate configurations this sweep compares: step-based, then
    altitude-based. Not a cross product -- the two forms are alternatives.

    Both lists are overridable so a follow-up pass can re-run one gate form
    without paying to retrain the other.
    """
    steps = STEP_THRESHOLDS if step_thresholds is None else step_thresholds
    altitudes = ALTITUDE_THRESHOLDS if altitude_thresholds is None else altitude_thresholds
    grid = [{"lockout_steps": s} for s in steps]
    grid += [{"altitude_threshold": a} for a in altitudes]
    return grid


def gate_label(gate_kwargs: Dict[str, float]) -> str:
    if "lockout_steps" in gate_kwargs:
        return f"step<{gate_kwargs['lockout_steps']}"
    return f"alt>{gate_kwargs['altitude_threshold']}"


def run_heuristic_leg(
    eval_episodes: int = 100,
    seed_start: int = HOLDOUT_SEED_START,
    grid: Optional[List[Dict[str, float]]] = None,
) -> pd.DataFrame:
    """No training involved -- the gate lives in the wrapper, so this just
    re-scores the heuristic with it active around evaluation, on the same
    held-out episodes used everywhere else in this repo."""
    rows = [
        {
            "gate": "none",
            **evaluate_controller_natural(
                HeuristicController(), eval_episodes, seed_start=seed_start
            ),
        }
    ]
    for gate_kwargs in (lockout_grid() if grid is None else grid):
        metrics = evaluate_controller_natural(
            HeuristicController(),
            eval_episodes,
            seed_start=seed_start,
            env_wrapper=lambda env, gk=gate_kwargs: EngineLockoutWrapper(env, **gk),
        )
        rows.append({"gate": gate_label(gate_kwargs), **gate_kwargs, **metrics})
    return pd.DataFrame(rows)


def _train_and_eval_rl_lockout(args: tuple) -> Dict[str, float]:
    """Train one PPO model with the lockout wrapper on the training env only
    (never eval -- same discipline as TimePenaltyWrapper), then score it.
    Runs in a worker process."""
    gate_kwargs, seed, total_timesteps, eval_episodes = args

    import torch

    # One thread per worker: torch defaults to 4, which would oversubscribe
    # the CPU several times over once the Pool is saturated.
    torch.set_num_threads(1)

    from ..controllers.rl_agent import RLAgent

    label = gate_label(gate_kwargs)
    safe_label = label.replace("<", "lt").replace(">", "gt").replace(".", "_")
    agent = RLAgent()
    saved_path = agent.train(
        total_timesteps=total_timesteps,
        save_path=f"ppo_lockout_{safe_label}_s{seed}",
        env_wrapper=lambda env: EngineLockoutWrapper(env, **gate_kwargs),
        hyperparams={"seed": seed, "verbose": 0},
    )
    metrics = evaluate_controller_natural(agent, eval_episodes, seed_start=HOLDOUT_SEED_START)
    return {"gate": label, "seed": seed, **gate_kwargs, "detail": saved_path, **metrics}


def run_rl_leg(
    seeds: List[int],
    total_timesteps: int = 1_000_000,
    eval_episodes: int = 100,
    n_jobs: Optional[int] = None,
    grid: Optional[List[Dict[str, float]]] = None,
) -> pd.DataFrame:
    """Train/evaluate the RL leg across the gate grid x `seeds`, pooled
    across workers the same way `time_penalty.run_multi_seed_sweep` is."""
    n_jobs = n_jobs or os.cpu_count() or 1
    work_items = [
        (gate_kwargs, seed, total_timesteps, eval_episodes)
        for gate_kwargs in (lockout_grid() if grid is None else grid)
        for seed in seeds
    ]
    print(
        f"Training {len(work_items)} PPO configs x {total_timesteps:,} timesteps "
        f"on {min(n_jobs, len(work_items))} workers..."
    )
    rows = []
    with multiprocessing.Pool(processes=min(n_jobs, len(work_items))) as pool:
        for i, result in enumerate(pool.imap_unordered(_train_and_eval_rl_lockout, work_items), 1):
            rows.append(result)
            print(
                f"  [{i}/{len(work_items)}] {result['gate']} seed={result['seed']} "
                f"reward={result['mean_reward']:.1f} steps={result['avg_landing_steps']:.1f}"
            )
    return pd.DataFrame(rows)


def _plot_lockout_results(heuristic_df: pd.DataFrame, rl_df: pd.DataFrame, path: Path) -> None:
    gate_order = ["none"] + [gate_label(g) for g in lockout_grid()]
    rl_by_gate = (
        rl_df.groupby("gate", sort=False)[["avg_landing_steps", "mean_reward"]]
        .mean()
        .reindex(gate_order)
    )
    heuristic_by_gate = heuristic_df.set_index("gate").reindex(gate_order)

    fig, (ax_steps, ax_reward) = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=_SURFACE)
    _style_axes(ax_steps, ax_reward)

    width = 0.35
    positions = range(len(gate_order))
    x_heuristic = list(positions)
    x_rl = [p + width for p in positions]

    for ax, column, title, ylabel in (
        (ax_steps, "avg_landing_steps", "Landing Speed by Lockout Gate",
         "avg landing steps (successful episodes)"),
        (ax_reward, "mean_reward", "Reward by Lockout Gate", "mean reward (natural, held-out)"),
    ):
        ax.bar(x_heuristic, heuristic_by_gate[column], width=width,
               color=_CONTROLLER_COLORS["Heuristic"], label="Heuristic")
        ax.bar(x_rl, rl_by_gate[column], width=width,
               color=_CONTROLLER_COLORS["RL (PPO)"], label="RL (PPO)")
        ax.set_xticks([p + width / 2 for p in positions], gate_order, rotation=30, ha="right")
        ax.set_ylabel(ylabel, color=_INK_PRIMARY)
        ax.set_title(title, color=_INK_PRIMARY)

    handles, labels = ax_steps.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06), frameon=False)
    fig.suptitle("Engine-Lockout Gating Sweep", color=_INK_PRIMARY, y=1.16)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=_SURFACE, bbox_inches="tight")
    plt.close(fig)


# Below this success rate, `avg_landing_steps` is averaged over so few
# surviving episodes that it measures which episodes got lucky, not speed.
# Those points are drawn hollow rather than dropped -- hiding them would hide
# the failure, but joining them to the solid line would imply comparability.
SURVIVORSHIP_SUCCESS_FLOOR = 95.0


def plot_step_sweep(
    heuristic_df: pd.DataFrame, rl_df: pd.DataFrame, path: Path, rl_baseline_note: str = ""
) -> None:
    """Two panels over the numeric lockout threshold: success rate, then
    landing steps. Separate panels rather than one dual-axis chart -- the two
    measures have unrelated scales, and overlaying them would invent a visual
    crossover that means nothing.
    """
    fig, (ax_success, ax_steps) = plt.subplots(
        1, 2, figsize=(12, 4.6), facecolor=_SURFACE, sharex=True
    )
    _style_axes(ax_success, ax_steps)

    # Both controllers are at 0% success by 100 steps and contribute no landing
    # time there, so plotting out to 100 spends half the width on one dead
    # point and compresses the region where every finding lives. Cut the axis
    # and say so on the figure rather than silently dropping data.
    x_max = 55

    for df, name in ((heuristic_df, "Heuristic"), (rl_df, "RL (PPO)")):
        if df is None or df.empty:
            continue
        d = df[df["lockout_steps"] <= x_max].sort_values("lockout_steps")
        color = _CONTROLLER_COLORS[name]
        ax_success.plot(d["lockout_steps"], d["success_rate_pct"],
                        marker="o", markersize=8, linewidth=2, color=color, label=name)

        # Solid line only through the trustworthy region; contaminated points
        # marked but not connected into it.
        trusted = d[d["success_rate_pct"] >= SURVIVORSHIP_SUCCESS_FLOOR]
        ax_steps.plot(trusted["lockout_steps"], trusted["avg_landing_steps"],
                      marker="o", markersize=8, linewidth=2, color=color, label=name)
        suspect = d[d["success_rate_pct"] < SURVIVORSHIP_SUCCESS_FLOOR]
        ax_steps.scatter(suspect["lockout_steps"], suspect["avg_landing_steps"],
                         s=64, facecolors="none", edgecolors=color, linewidths=1.8, zorder=3)

    ax_success.set_ylabel("success rate (%)", color=_INK_PRIMARY)
    ax_success.set_title("Success vs. Lockout Length", color=_INK_PRIMARY)
    ax_success.set_ylim(-3, 105)

    ax_steps.set_ylabel("avg landing steps (successful episodes)", color=_INK_PRIMARY)
    ax_steps.set_title("Landing Speed vs. Lockout Length", color=_INK_PRIMARY)

    for ax in (ax_success, ax_steps):
        ax.set_xlabel("main-engine lockout (steps from episode start)", color=_INK_PRIMARY)
        ax.set_xlim(-3, x_max)

    handles, labels = ax_success.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.05), frameon=False)
    fig.suptitle("Engine-Lockout Sweep: where a free win becomes a collapse",
                 color=_INK_PRIMARY, y=1.14)
    fig.text(
        0.5, -0.09,
        f"x=0 is no lockout. Axis cut at {x_max}: both controllers are at 0% success by 100 "
        f"steps and record no landing time there.\n"
        f"Hollow markers: success < {SURVIVORSHIP_SUCCESS_FLOOR:.0f}%, so the step average is "
        f"over few surviving episodes and reads fast by selection, not by speed."
        f"{rl_baseline_note}",
        ha="center", fontsize=8, color=_INK_MUTED,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=_SURFACE, bbox_inches="tight")
    plt.close(fig)


def run_lockout_sweep(
    total_timesteps: int = 1_000_000,
    eval_episodes: int = 100,
    seeds: Optional[List[int]] = None,
    n_jobs: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Run both legs of the sweep and write CSVs + a comparison plot.

    `seeds` defaults to a single-seed first pass (`[0]`) -- pass the winning
    config's own 3 seeds for the roadmap's confirm-on-the-winner pass.
    """
    seeds = seeds if seeds is not None else [0]
    sweep_dir = Path(output_dir) if output_dir else new_run_dir("lockout_sweep")
    sweep_dir.mkdir(parents=True, exist_ok=True)

    print("=== Heuristic leg ===")
    heuristic_df = run_heuristic_leg(eval_episodes=eval_episodes)
    heuristic_df.to_csv(sweep_dir / "heuristic_results.csv", index=False)
    print(heuristic_df.to_string(index=False))

    print("\n=== RL leg ===")
    rl_df = run_rl_leg(
        seeds=seeds, total_timesteps=total_timesteps, eval_episodes=eval_episodes, n_jobs=n_jobs
    )
    rl_df.to_csv(sweep_dir / "rl_results.csv", index=False)
    print(rl_df.to_string(index=False))

    plot_path = sweep_dir / "lockout_tradeoff.png"
    _plot_lockout_results(heuristic_df, rl_df, plot_path)

    print(f"\nHeuristic CSV: {sweep_dir / 'heuristic_results.csv'}")
    print(f"RL CSV:        {sweep_dir / 'rl_results.csv'}")
    print(f"Plot:          {plot_path}")

    return {"heuristic": heuristic_df, "rl": rl_df}
