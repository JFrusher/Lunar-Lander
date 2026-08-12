"""Regenerate the curated figures for the speed-optimization arc into docs/media/.

`runs/` is gitignored working data; `docs/media/` is the small tracked set that
README.md and INVESTIGATION.md actually link. This script rebuilds the latter
from the former so the tracked figures are reproducible rather than hand-copied.

Usage:  python scripts/regen_speed_media.py
"""

import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lunar_lander_lab.utils.lockout_sweep import (  # noqa: E402
    lockout_grid,
    plot_step_sweep,
    run_heuristic_leg,
)
from lunar_lander_lab.utils.speed_ceiling import run_speed_ceiling  # noqa: E402

MEDIA = ROOT / "docs" / "media"

# Phase 2, RL leg. Assembled from three runs rather than one: the first pass
# covered 20/50/100, the finer-grid pass added 5/10/15/30/40, and the 3-seed
# confirms cover 10/15/20. A fixed seed reproduces, so nothing was re-run.
RL_SEED0 = {
    0: (257.938144, 99.0, 334.383838),   # baseline, penalty=0.0, seed 0
    5: (260.333819, 99.0, 327.262626),
    10: (265.556637, 97.0, 249.989691),
    15: (263.540456, 95.0, 244.747368),
    20: (268.290951, 98.0, 264.408163),
    30: (238.150199, 96.0, 410.145833),
    40: (35.295993, 1.0, 674.000000),
    50: (89.972220, 6.0, 756.000000),
    100: (-2286.006003, 0.0, float("nan")),
}

def _heuristic_frame() -> pd.DataFrame:
    """Re-run the heuristic leg rather than reading a cached CSV.

    It's a deterministic controller on a fixed held-out episode set, so this
    reproduces bit-for-bit, costs a couple of minutes, and keeps the script
    runnable from a fresh clone -- `runs/` and `tmp/` are both gitignored, so
    anything cached there would not survive one.
    """
    df = run_heuristic_leg(eval_episodes=100, grid=lockout_grid(altitude_thresholds=[]))
    # "none" is the no-lockout baseline; on a numeric threshold axis that's 0.
    df["lockout_steps"] = df["lockout_steps"].fillna(0)
    return df[["lockout_steps", "mean_reward", "success_rate_pct", "avg_landing_steps"]]


def _rl_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"lockout_steps": k, "mean_reward": v[0],
             "success_rate_pct": v[1], "avg_landing_steps": v[2]}
            for k, v in RL_SEED0.items()
        ]
    )


# Phase 10 rollup. One consistent 100-held-out-episode run at nominal physics
# (runs/robustness/20260812_073352), so every point is directly comparable --
# and `robust` is mean success across the five off-nominal conditions.
ROLLUP = [
    # name, flight steps, success %, off-nominal mean success %, family
    ("Heuristic\n(pre-Phase-4)", 329.2, 98.0, 50.2, "Heuristic"),
    ("Heuristic\n(flat)", 301.1, 100.0, 51.2, "Heuristic"),
    ("Heuristic\n+ lockout 15", 277.1, 99.0, 47.0, "Heuristic"),
    ("PPO\ncurriculum", 200.7, 99.0, 54.6, "RL (PPO)"),
    ("PPO\nlockout-trained", 186.1, 95.0, 66.2, "RL (PPO)"),
    # Arc 3
    ("Gain-scheduled\nheuristic", 185.9, 100.0, 47.8, "Heuristic"),
    ("MPC\n(planner)", 157.8, 72.0, 30.6, "Planner"),
]

# Phase 1's bound, evaluated at the touchdown speed actually measured
# (1.42 u/s). A FLIGHT-time bound -- it stops at ground contact and knows
# nothing about the settling tail, which is why this chart plots flight steps.
CEILING_FLIGHT_TICKS = 124.0


def _plot_frontier(path: Path) -> None:
    """Speed vs reliability across every technique, with the physical bound.

    Marker size encodes off-nominal robustness, so the chart shows all three
    quantities that turned out to matter without a second panel or a colour
    ramp competing with the categorical controller colours.
    """
    from lunar_lander_lab.utils.robustness import pareto_frontier
    from lunar_lander_lab.utils.time_penalty import (
        _CONTROLLER_COLORS,
        _GRID,
        _INK_MUTED,
        _INK_PRIMARY,
        _SURFACE,
    )

    # Arc 3 adds a third controller family. Slot 3 of the same categorical
    # theme, validated with the dataviz palette checker: it does not worsen
    # the worst CVD-adjacent pair, which remains the original blue/orange at
    # deltaE 24.7 (protan).
    families = {**_CONTROLLER_COLORS, "Planner": "#7a4fc0"}

    fig, ax = plt.subplots(figsize=(10, 5.6), facecolor=_SURFACE)
    ax.set_facecolor(_SURFACE)
    ax.grid(True, color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(_GRID)
    ax.tick_params(colors=_INK_MUTED)

    pts = [{"name": n, "flight_steps": f, "success_rate_pct": s} for n, f, s, _, _ in ROLLUP]
    on_frontier = {p["name"] for p in pareto_frontier(pts)}

    ax.axvline(CEILING_FLIGHT_TICKS, color=_INK_MUTED, linestyle="--", linewidth=1.4)
    ax.text(CEILING_FLIGHT_TICKS + 4, 69.0,
            f"idealized floor\n{CEILING_FLIGHT_TICKS:.0f} ticks",
            fontsize=8, color=_INK_MUTED, va="bottom")

    frontier_pts = sorted(
        [(f, s) for n, f, s, _, _ in ROLLUP if n in on_frontier], key=lambda t: t[0]
    )
    ax.plot([f for f, _ in frontier_pts], [s for _, s in frontier_pts],
            color=_INK_MUTED, linewidth=1.2, alpha=0.5, zorder=1)

    for name, flight, success, robust, family in ROLLUP:
        ax.scatter(
            flight, success,
            s=40 + robust * 9,
            color=families[family],
            edgecolors=_INK_PRIMARY if name in on_frontier else "none",
            linewidths=1.6, zorder=3, alpha=0.9,
        )
        ax.annotate(name, (flight, success), textcoords="offset points",
                    xytext=(0, -34), ha="center", fontsize=8, color=_INK_PRIMARY)

    ax.set_xlabel("flight steps to first ground contact (lower is faster)", color=_INK_PRIMARY)
    ax.set_ylabel("success rate at nominal physics (%)", color=_INK_PRIMARY)
    ax.set_xlim(120, 360)
    ax.set_ylim(68, 103)
    ax.set_title("Every technique tried: speed, reliability, and transfer",
                 color=_INK_PRIMARY, fontsize=12, pad=14)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", markersize=9,
                   color=families[f], label=f)
        for f in ("Heuristic", "RL (PPO)", "Planner")
    ]
    handles.append(plt.Line2D([], [], marker="o", linestyle="none", markersize=11,
                              markerfacecolor="none", markeredgecolor=_INK_PRIMARY,
                              label="on Pareto frontier"))
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)

    fig.text(0.5, -0.03,
             "Marker size = mean success across five off-nominal physics conditions "
             "(bigger transfers better). Flight steps, not totals: the bound is a "
             "flight-time bound.",
             ha="center", fontsize=8, color=_INK_MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=_SURFACE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    MEDIA.mkdir(parents=True, exist_ok=True)

    print("=== Phase 1: speed ceiling ===")
    run_speed_ceiling()
    newest = max((ROOT / "runs" / "speed_ceiling").iterdir(), key=lambda p: p.name)
    shutil.copy2(newest / "speed_ceiling.png", MEDIA / "speed_ceiling.png")
    print(f"  -> {MEDIA / 'speed_ceiling.png'}")

    print("\n=== Phase 2: lockout step sweep ===")
    plot_step_sweep(
        _heuristic_frame(),
        _rl_frame(),
        MEDIA / "lockout_step_sweep.png",
        rl_baseline_note=" RL curve is seed 0 throughout, including its x=0 baseline.",
    )
    print(f"  -> {MEDIA / 'lockout_step_sweep.png'}")

    print("\n=== Phase 10: rollup frontier ===")
    _plot_frontier(MEDIA / "frontier.png")
    print(f"  -> {MEDIA / 'frontier.png'}")


if __name__ == "__main__":
    main()
