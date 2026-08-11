"""Phase 5 of tmp/SPEED_ROADMAP.md: anneal the time penalty during training
instead of fixing it at step 0."""

import gymnasium as gym
import pytest

from lunar_lander_lab.utils.penalty_curriculum import (
    SCHEDULES,
    CurriculumTimePenaltyWrapper,
    linear_schedule,
    stepped_schedule,
)

TOTAL = 1_000_000


def test_linear_schedule_ramps_from_zero_to_the_target():
    assert linear_schedule(0, TOTAL, 0.4) == pytest.approx(0.0)
    assert linear_schedule(TOTAL // 2, TOTAL, 0.4) == pytest.approx(0.2)
    assert linear_schedule(TOTAL, TOTAL, 0.4) == pytest.approx(0.4)


def test_linear_schedule_clamps_past_the_budget():
    """Training can overrun the nominal budget by a partial rollout; the
    penalty must not keep climbing past its target when it does."""
    assert linear_schedule(TOTAL * 2, TOTAL, 0.4) == pytest.approx(0.4)


def test_stepped_schedule_walks_the_measured_levels_in_order():
    """The levels are old roadmap Phase 4's own sweep points, so the run is
    directly comparable to the flat-penalty results at each one."""
    levels = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]
    seen = [stepped_schedule(int(TOTAL * f), TOTAL, levels) for f in
            (0.0, 0.2, 0.4, 0.6, 0.8, 0.99)]

    assert seen == levels


def test_stepped_schedule_holds_the_top_level_at_and_past_the_end():
    levels = [0.0, 0.1, 0.4]
    assert stepped_schedule(TOTAL, TOTAL, levels) == pytest.approx(0.4)
    assert stepped_schedule(TOTAL * 3, TOTAL, levels) == pytest.approx(0.4)


def test_wrapper_step_count_persists_across_episodes():
    """The whole mechanism depends on this. The flat wrapper is stateless per
    episode; a curriculum has to count total training steps, so `reset()`
    must NOT zero the counter -- if it did, the penalty would restart at 0.0
    every episode and no annealing would ever happen."""
    env = CurriculumTimePenaltyWrapper(
        gym.make("LunarLander-v3"), schedule_fn=lambda t: float(t)
    )
    env.reset(seed=0)
    for _ in range(5):
        env.step(0)
    env.reset(seed=1)
    for _ in range(3):
        env.step(0)
    env.close()

    assert env.total_steps == 8


def test_wrapper_subtracts_the_scheduled_penalty_for_the_current_step():
    plain = gym.make("LunarLander-v3")
    # Constant schedule so the expected deduction is unambiguous.
    shaped = CurriculumTimePenaltyWrapper(
        gym.make("LunarLander-v3"), schedule_fn=lambda _t: 0.25
    )
    plain.reset(seed=0)
    shaped.reset(seed=0)

    _, reward_plain, *_ = plain.step(0)
    _, reward_shaped, *_ = shaped.step(0)
    plain.close()
    shaped.close()

    assert reward_plain - 0.25 == pytest.approx(reward_shaped)


def test_wrapper_records_the_penalty_it_actually_reached():
    """The headline of this phase is 'how far up the penalty did it get
    before collapsing', so the reached level has to be readable afterwards.

    Invariant: `current_penalty` always equals `schedule_fn(total_steps)` --
    the counter is advanced first, so the two never disagree about where in
    training the run is.
    """
    schedule = lambda t: t * 0.001  # noqa: E731
    env = CurriculumTimePenaltyWrapper(gym.make("LunarLander-v3"), schedule_fn=schedule)
    env.reset(seed=0)
    for _ in range(10):
        env.step(0)
    env.close()

    assert env.total_steps == 10
    assert env.current_penalty == pytest.approx(schedule(env.total_steps))
    assert env.current_penalty == pytest.approx(0.010)


def test_named_schedules_all_start_at_zero_and_reach_their_target():
    """Every shape must share the same endpoints, or the comparison between
    them is confounded by where they start and finish rather than by shape."""
    for name, build in SCHEDULES.items():
        fn = build(TOTAL, 0.4)
        assert fn(0) == pytest.approx(0.0), name
        assert fn(TOTAL) == pytest.approx(0.4), name
