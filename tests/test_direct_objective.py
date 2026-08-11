"""Phase 4 of tmp/SPEED_ROADMAP.md: rank gain sets by speed subject to a
success floor, instead of by a reward-minus-penalty blend."""

import numpy as np
import pandas as pd
import pytest

from lunar_lander_lab.utils.pid_search import select_fastest_within_floor


def _frame(rows):
    return pd.DataFrame(
        rows, columns=["name", "success_rate_pct", "avg_steps_success", "mean_reward"]
    )


def test_ranks_by_speed_not_by_reward():
    """The whole point of the phase: the reward-optimal gain set and the
    fastest safe one need not be the same, and the existing search only ever
    looked for the former."""
    df = _frame([
        ("slow_high_reward", 100.0, 400.0, 280.0),
        ("fast_lower_reward", 100.0, 250.0, 240.0),
    ])

    winners = select_fastest_within_floor(df, success_floor=95.0)

    assert list(winners["name"]) == ["fast_lower_reward", "slow_high_reward"]


def test_excludes_rows_under_the_success_floor():
    """A 200-step lander that only lands 40% of the time is not a faster
    controller, it's a broken one that got lucky on the episodes it survived."""
    df = _frame([
        ("very_fast_unsafe", 40.0, 200.0, 100.0),
        ("safe", 97.0, 380.0, 250.0),
    ])

    winners = select_fastest_within_floor(df, success_floor=95.0)

    assert list(winners["name"]) == ["safe"]


def test_excludes_rows_that_never_landed():
    """avg_steps_success is NaN when nothing landed -- those must not sort to
    the top as if they were infinitely fast."""
    df = _frame([
        ("never_landed", 0.0, np.nan, -100.0),
        ("safe", 99.0, 350.0, 250.0),
    ])

    winners = select_fastest_within_floor(df, success_floor=0.0)

    assert list(winners["name"]) == ["safe"]


def test_respects_top_k():
    df = _frame([(f"g{i}", 100.0, 300.0 + i, 250.0) for i in range(20)])

    assert len(select_fastest_within_floor(df, success_floor=95.0, top_k=5)) == 5


def test_returns_empty_when_nothing_clears_the_floor():
    """A floor no sample meets is a real answer -- 'the pool has nothing this
    safe' -- not an error, and not a silently relaxed floor."""
    df = _frame([("a", 50.0, 300.0, 100.0), ("b", 60.0, 320.0, 120.0)])

    winners = select_fastest_within_floor(df, success_floor=99.0)

    assert len(winners) == 0


@pytest.mark.parametrize("floor", [95.0, 99.0])
def test_every_returned_row_meets_the_floor(floor):
    rng = np.random.default_rng(0)
    df = _frame([
        (f"g{i}", float(rng.uniform(0, 100)), float(rng.uniform(200, 500)), 0.0)
        for i in range(200)
    ])

    winners = select_fastest_within_floor(df, success_floor=floor)

    assert (winners["success_rate_pct"] >= floor).all()
