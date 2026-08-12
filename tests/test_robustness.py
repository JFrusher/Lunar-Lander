"""Phase 10 of tmp/SPEED_ROADMAP.md: does a controller tuned at nominal
physics survive conditions it never saw?"""

import gymnasium as gym
import pandas as pd
import pytest

from lunar_lander_lab.utils.robustness import (
    CONDITIONS,
    condition_labels,
    pareto_frontier,
    summarise_transfer,
)


def _p(name, flight, success):
    return {"name": name, "flight_steps": flight, "success_rate_pct": success}


def test_pareto_drops_points_beaten_on_both_axes():
    points = [_p("fast_safe", 180, 99.0), _p("slow_risky", 300, 90.0)]

    assert [p["name"] for p in pareto_frontier(points)] == ["fast_safe"]


def test_pareto_keeps_genuine_tradeoffs():
    """Faster-but-riskier and slower-but-safer are both on the frontier --
    which of them is 'best' depends on a success floor the roadmap
    deliberately never fixed."""
    points = [_p("fast_risky", 180, 95.0), _p("slow_safe", 300, 100.0)]

    assert [p["name"] for p in pareto_frontier(points)] == ["fast_risky", "slow_safe"]


def test_pareto_returns_sorted_by_speed():
    # All three are genuine trade-offs (each faster one is less reliable), so
    # all survive -- and must come back fastest-first regardless of input order.
    points = [_p("c", 300, 100.0), _p("a", 180, 95.0), _p("b", 240, 98.0)]

    assert [p["name"] for p in pareto_frontier(points)] == ["a", "b", "c"]


def test_pareto_keeps_exact_duplicates():
    """A tie must not silently delete both points (or one arbitrarily)."""
    points = [_p("x", 200, 98.0), _p("y", 200, 98.0)]

    assert len(pareto_frontier(points)) == 2


def test_nominal_is_first_and_empty():
    """The nominal row is the reference every other row is measured against,
    so it must be the unmodified environment."""
    label, kwargs = CONDITIONS[0]
    assert label == "nominal"
    assert kwargs == {}


def test_every_condition_is_constructible():
    """gravity must sit strictly inside (-12, 0) -- a round -12.0 raises, and
    an unconstructible condition would fail an hour into an eval sweep."""
    for label, kwargs in CONDITIONS:
        env = gym.make("LunarLander-v3", **kwargs)
        try:
            env.reset(seed=0)
        finally:
            env.close()


def test_condition_labels_are_unique():
    assert len(condition_labels()) == len(set(condition_labels()))


def test_summarise_transfer_reports_the_worst_case_not_the_average():
    """A controller that holds up in four conditions and collapses in the
    fifth is not robust, and an average would hide that."""
    df = pd.DataFrame(
        [
            {"controller": "steady", "condition": "nominal", "success_rate_pct": 99.0},
            {"controller": "steady", "condition": "wind 10", "success_rate_pct": 95.0},
            {"controller": "steady", "condition": "wind 15", "success_rate_pct": 94.0},
            {"controller": "brittle", "condition": "nominal", "success_rate_pct": 100.0},
            {"controller": "brittle", "condition": "wind 10", "success_rate_pct": 97.0},
            {"controller": "brittle", "condition": "wind 15", "success_rate_pct": 10.0},
        ]
    )

    summary = summarise_transfer(df).set_index("controller")

    assert summary.loc["brittle", "worst_success"] == 10.0
    assert summary.loc["brittle", "success_drop"] == 90.0
    assert summary.loc["steady", "success_drop"] == 5.0
    # Ranked most-robust-first, so the brittle one cannot hide at the top.
    assert list(summarise_transfer(df)["controller"]) == ["steady", "brittle"]


def test_summarise_transfer_excludes_nominal_from_the_worst_case():
    """Nominal is the reference; including it could only ever mask a drop."""
    df = pd.DataFrame(
        [
            {"controller": "a", "condition": "nominal", "success_rate_pct": 50.0},
            {"controller": "a", "condition": "wind 10", "success_rate_pct": 80.0},
        ]
    )

    summary = summarise_transfer(df)

    assert summary.loc[0, "worst_success"] == 80.0
    assert summary.loc[0, "success_drop"] == pytest.approx(-30.0)
