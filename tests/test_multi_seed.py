import numpy as np
import pandas as pd

from lunar_lander_lab.utils.time_penalty import aggregate_seeds


def _raw(rows):
    return pd.DataFrame(
        [
            {
                "controller": c,
                "penalty": p,
                "seed": s,
                "mean_reward": r,
                "success_rate_pct": sr,
                "crash_rate_pct": 0.0,
                "avg_landing_steps": steps,
            }
            for c, p, s, r, sr, steps in rows
        ]
    )


def test_aggregate_produces_the_columns_the_plot_reads():
    agg = aggregate_seeds(
        _raw([("Heuristic", 0.0, s, 250.0, 100.0, 400.0) for s in (0, 100, 200)])
    )
    for metric in ("mean_reward", "success_rate_pct", "crash_rate_pct", "avg_landing_steps"):
        assert f"{metric}_mean" in agg.columns
        assert f"{metric}_std" in agg.columns
    assert {"controller", "penalty"} <= set(agg.columns)


def test_mean_and_std_are_computed_per_controller_and_penalty():
    agg = aggregate_seeds(
        _raw(
            [
                ("Heuristic", 0.0, 0, 240.0, 100.0, 400.0),
                ("Heuristic", 0.0, 100, 260.0, 100.0, 400.0),
                ("Heuristic", 0.2, 0, 100.0, 50.0, 300.0),
                ("Heuristic", 0.2, 100, 100.0, 50.0, 300.0),
                ("RL (PPO)", 0.0, 0, 10.0, 5.0, 700.0),
                ("RL (PPO)", 0.0, 100, 30.0, 5.0, 700.0),
            ]
        )
    )
    assert len(agg) == 3

    row = agg[(agg.controller == "Heuristic") & (agg.penalty == 0.0)].iloc[0]
    assert row["mean_reward_mean"] == 250.0
    assert row["mean_reward_std"] == pd.Series([240.0, 260.0]).std()

    # Identical seeds must report zero spread, not NaN — a flat error bar is a
    # real result, and NaN would silently vanish from the chart.
    flat = agg[(agg.controller == "Heuristic") & (agg.penalty == 0.2)].iloc[0]
    assert flat["mean_reward_std"] == 0.0

    rl = agg[agg.controller == "RL (PPO)"].iloc[0]
    assert rl["mean_reward_mean"] == 20.0


def test_single_seed_gives_nan_std_not_a_crash():
    """One seed can't estimate a spread; NaN is the honest answer."""
    agg = aggregate_seeds(_raw([("Heuristic", 0.0, 0, 250.0, 100.0, 400.0)]))
    assert np.isnan(agg.iloc[0]["mean_reward_std"])
    assert agg.iloc[0]["mean_reward_mean"] == 250.0
