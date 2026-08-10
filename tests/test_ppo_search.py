import pandas as pd

from lunar_lander_lab.utils.pid_search import sample_gain_sets
from lunar_lander_lab.utils.ppo_search import PPO_PARAM_SPACE, to_hyperparams


def test_to_hyperparams_respects_bounds():
    for sample in sample_gain_sets(50, PPO_PARAM_SPACE, seed=0):
        hp = to_hyperparams(sample)
        assert 1e-4 <= hp["learning_rate"] <= 1e-3
        assert 512 - 64 <= hp["n_steps"] <= 4096 + 64
        assert 0.0 <= hp["ent_coef"] <= 0.05
        assert 0.99 <= hp["gamma"] <= 0.999
        assert 0.9 <= hp["gae_lambda"] <= 0.99


def test_n_steps_is_whole_minibatches():
    for sample in sample_gain_sets(50, PPO_PARAM_SPACE, seed=1):
        n_steps = to_hyperparams(sample)["n_steps"]
        assert isinstance(n_steps, int)
        assert n_steps % 64 == 0
        assert n_steps >= 64


def test_learning_rate_is_log_uniform():
    """Half the samples should land below the geometric midpoint (~3.16e-4).

    A linear sweep of 1e-4..1e-3 would put ~76% of samples above it.
    """
    lrs = [to_hyperparams(s)["learning_rate"] for s in sample_gain_sets(200, PPO_PARAM_SPACE, seed=2)]
    below = sum(lr < 10 ** -3.5 for lr in lrs)
    assert 0.4 <= below / len(lrs) <= 0.6, below / len(lrs)


def test_ranking_picks_highest_mean_reward():
    df = pd.DataFrame(
        {
            "learning_rate": [1e-4, 5e-4, 9e-4],
            "mean_reward": [100.0, 280.0, 40.0],
        }
    ).sort_values("mean_reward", ascending=False)
    assert df.iloc[0]["learning_rate"] == 5e-4
