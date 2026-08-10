import pandas as pd

from lunar_lander_lab.utils.ppo_convergence import recommend_timesteps


def test_recommend_timesteps_picks_earliest_plateau():
    df = pd.DataFrame(
        {
            "total_timesteps": [100_000, 200_000, 400_000, 700_000, 1_000_000],
            "mean_reward": [50.0, 240.0, 245.0, 248.0, 250.0],
            "success_rate_pct": [10.0, 96.0, 96.0, 98.0, 100.0],
        }
    )
    assert recommend_timesteps(df) == 200_000


def test_recommend_timesteps_requires_reward_near_peak():
    df = pd.DataFrame(
        {
            "total_timesteps": [100_000, 200_000, 400_000],
            "mean_reward": [50.0, 100.0, 250.0],
            # success rate alone would falsely qualify 200_000 without the
            # reward-fraction check, since it's within tolerance of the peak.
            "success_rate_pct": [10.0, 97.0, 100.0],
        }
    )
    assert recommend_timesteps(df) == 400_000


def test_recommend_timesteps_single_row():
    df = pd.DataFrame(
        {
            "total_timesteps": [500_000],
            "mean_reward": [200.0],
            "success_rate_pct": [90.0],
        }
    )
    assert recommend_timesteps(df) == 500_000
