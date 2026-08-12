"""Do the fast controllers survive conditions they were never tuned on?

Every result in this project was produced at the environment's defaults:
`gravity=-10.0`, no wind. A controller tuned hard against one fixed physics
setting can be fast because it fits that setting, not because it flies well.
Phase 10 of tmp/SPEED_ROADMAP.md.

Evaluation only -- nothing is retrained. That is the point: the question is
whether a controller selected under nominal physics *transfers*, not whether
it could be re-tuned to succeed.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .evaluation import evaluate_controller_natural
from .paths import new_run_dir
from .pid_search import HOLDOUT_SEED_START

# gravity must sit strictly inside (-12, 0), so the harsh end is -11.9 rather
# than a round -12. Roughly +/-20% around the -10.0 default.
NOMINAL_GRAVITY = -10.0

CONDITIONS: List[Tuple[str, Dict]] = [
    ("nominal", {}),
    ("gravity -8.0 (weak)", {"gravity": -8.0}),
    ("gravity -11.9 (strong)", {"gravity": -11.9}),
    ("wind 10", {"enable_wind": True, "wind_power": 10.0}),
    ("wind 15", {"enable_wind": True, "wind_power": 15.0}),
    ("gravity -11.9 + wind 15", {"gravity": -11.9, "enable_wind": True, "wind_power": 15.0}),
]


def condition_labels() -> List[str]:
    return [label for label, _ in CONDITIONS]


def summarise_transfer(df: pd.DataFrame) -> pd.DataFrame:
    """Per controller: nominal success, worst-case success, and the drop.

    The drop is the number that matters. A controller that is fast at nominal
    and falls apart under a 20% gravity change was fitted to the physics, not
    to the task.
    """
    rows = []
    for name, group in df.groupby("controller", sort=False):
        nominal = group.loc[group["condition"] == "nominal", "success_rate_pct"]
        nominal_success = float(nominal.iloc[0]) if len(nominal) else float("nan")
        off_nominal = group[group["condition"] != "nominal"]["success_rate_pct"]
        worst = float(off_nominal.min()) if len(off_nominal) else float("nan")
        rows.append(
            {
                "controller": name,
                "nominal_success": nominal_success,
                "worst_success": worst,
                "success_drop": nominal_success - worst,
                "mean_off_nominal_success": float(off_nominal.mean()) if len(off_nominal) else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("success_drop")


def pareto_frontier(points: List[Dict]) -> List[Dict]:
    """Points not beaten on both speed and success by some other point.

    Speed is `flight_steps` (lower better) and reliability is
    `success_rate_pct` (higher better). Flight rather than total steps, because
    the idealized bound these are compared against is a flight-time bound and
    ~23% of a total is post-touchdown settling no controller can influence.

    Ties do not eliminate: a point is dropped only if another is strictly
    better on one axis and at least equal on the other, so duplicates and
    equal-performing variants both survive.
    """
    frontier = []
    for candidate in points:
        dominated = any(
            other is not candidate
            and other["flight_steps"] <= candidate["flight_steps"]
            and other["success_rate_pct"] >= candidate["success_rate_pct"]
            and (
                other["flight_steps"] < candidate["flight_steps"]
                or other["success_rate_pct"] > candidate["success_rate_pct"]
            )
            for other in points
        )
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda p: p["flight_steps"])


def run_robustness_check(
    controllers: Dict[str, object],
    eval_episodes: int = 100,
    seed_start: int = HOLDOUT_SEED_START,
    env_wrappers: Optional[Dict[str, object]] = None,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Score every controller under every condition on identical episodes."""
    env_wrappers = env_wrappers or {}
    out_dir = Path(output_dir) if output_dir else new_run_dir("robustness")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, controller in controllers.items():
        for label, env_kwargs in CONDITIONS:
            metrics = evaluate_controller_natural(
                controller,
                eval_episodes,
                seed_start=seed_start,
                env_kwargs=env_kwargs,
                env_wrapper=env_wrappers.get(name),
            )
            rows.append({"controller": name, "condition": label, **metrics})
            print(f"  {name:28s} {label:26s} "
                  f"success={metrics['success_rate_pct']:5.1f}% "
                  f"reward={metrics['mean_reward']:7.1f} "
                  f"flight={metrics['avg_flight_steps']:6.1f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "robustness_raw.csv", index=False)
    summary = summarise_transfer(df)
    summary.to_csv(out_dir / "robustness_summary.csv", index=False)

    print(f"\nRaw:     {out_dir / 'robustness_raw.csv'}")
    print(f"Summary: {out_dir / 'robustness_summary.csv'}")
    print(summary.to_string(index=False))
    return df
