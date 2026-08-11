"""Regenerate the curated figures for the speed-optimization arc into docs/media/.

`runs/` is gitignored working data; `docs/media/` is the small tracked set that
README.md and INVESTIGATION.md actually link. This script rebuilds the latter
from the former so the tracked figures are reproducible rather than hand-copied.

Usage:  python scripts/regen_speed_media.py
"""

import shutil
import sys
from pathlib import Path

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


if __name__ == "__main__":
    main()
