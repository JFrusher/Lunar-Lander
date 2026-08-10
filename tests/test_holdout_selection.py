"""The winner's-curse fix: winners are re-scored on episodes the search never saw."""

import json

from lunar_lander_lab.utils.pid_search import (
    CORE_PARAM_SPACE,
    HOLDOUT_SEED_START,
    _evaluate_gain_set,
    run_monte_carlo,
    sample_gain_sets,
)

_GAINS = sample_gain_sets(1, CORE_PARAM_SPACE, seed=0)[0]


def test_holdout_episodes_cannot_overlap_the_search_episodes():
    """The search ranks on episodes 0..episodes_per_set-1; held-out must start past that."""
    assert HOLDOUT_SEED_START > 1000, "too close to plausible episodes_per_set values"


def test_seed_start_actually_changes_which_episodes_run():
    """If seed_start were ignored, the fix would be cosmetic and silently useless."""
    on_search = _evaluate_gain_set((_GAINS, 3, "LunarLander-v3", 0))
    on_holdout = _evaluate_gain_set((_GAINS, 3, "LunarLander-v3", HOLDOUT_SEED_START))
    assert on_search["mean_reward"] != on_holdout["mean_reward"]


def test_seed_start_is_reproducible():
    a = _evaluate_gain_set((_GAINS, 3, "LunarLander-v3", HOLDOUT_SEED_START))
    b = _evaluate_gain_set((_GAINS, 3, "LunarLander-v3", HOLDOUT_SEED_START))
    # Compared field-by-field rather than dict-to-dict: avg_steps_success is
    # NaN when a gain set never lands, and NaN != NaN.
    assert a["mean_reward"] == b["mean_reward"]
    assert a["success_rate_pct"] == b["success_rate_pct"]


def test_run_monte_carlo_reports_holdout_metrics(tmp_path):
    run_monte_carlo(
        n_samples=4,
        episodes_per_set=2,
        seed=0,
        output_dir=str(tmp_path),
        n_jobs=2,
        holdout_top_k=2,
        holdout_episodes=2,
    )

    summary = json.loads((tmp_path / "best_gains.json").read_text())
    assert summary["holdout_top_k"] == 2
    assert summary["holdout_episodes"] == 2
    # Both the corrected number and what the old selection would have claimed,
    # so the optimism stays visible instead of being silently corrected away.
    assert "search_set_best_mean_reward" in summary
    assert summary["search_set_optimism"] == (
        summary["search_set_best_mean_reward"] - summary["mean_reward"]
    )
    assert (tmp_path / "holdout_top_k.csv").exists()


def test_holdout_top_k_is_clamped_to_sample_count(tmp_path):
    run_monte_carlo(
        n_samples=3,
        episodes_per_set=2,
        seed=0,
        output_dir=str(tmp_path),
        n_jobs=2,
        holdout_top_k=99,
        holdout_episodes=2,
    )
    assert json.loads((tmp_path / "best_gains.json").read_text())["holdout_top_k"] == 3
