"""A modular marking framework: segment a descent, grade each part, name the
behaviours.

Every substantial finding across three arcs came from decomposing a scalar.
Total landing steps hid the fact that ~23% of them are incompressible
settling (Arc 2 Phase 6). One reward number hid that a time penalty saturates
(Arc 2 Phase 5). One gain set hid that the descent has two regimes (Arc 3
Phase C). This module generalises that move.

Three layers, deliberately separated so the analysis is pure and testable
without an environment:

1. `record_episode` -- runs one episode, returns a per-step frame. The only
   part that touches Gymnasium.
2. `summarise_segments` / `behaviour_scores` -- pure functions over that
   frame.
3. `report_card` -- grades segments against the *idealized plan run from the
   same start state and segmented the same way*, so the comparison is
   segment-to-segment rather than against an arbitrary target.

**Extension point for training** (designed, not built): the segment
boundaries here are exactly what a segment-aware reward wrapper needs. Arc 2
Phase 6 showed a flat time penalty charges for the settling tail, which no
policy can shorten -- pressure spent where the controller has no authority.
A penalty applied only to `DESCENT`/`APPROACH`/`TERMINAL` is the natural test
of whether that is why the penalty axis saturated.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium.envs.box2d.lunar_lander import SCALE, VIEWPORT_H

from ..controllers.base import BaseController
from ..controllers.scheduled_heuristic import BAND_EDGES
from .evaluation import (
    MAIN_ENGINE_REWARD_COST,
    SIDE_ENGINE_REWARD_COST,
    VY_NORM_TO_WORLD,
    count_engine_frames,
)
from .speed_ceiling import (
    DescentConstants,
    measure_descent_constants,
    min_time_to_land,
)

# Flight is subdivided by the same altitude bands the scheduled controller
# uses, so a poor segment grade points directly at the gains that produced
# it. Settling is event-based (first leg contact), because that boundary is
# physical rather than positional.
SEGMENTS: List[str] = ["DESCENT", "APPROACH", "TERMINAL", "SETTLING"]

# observation[1] is altitude normalised by this; the descent model is in world units.
Y_WORLD_PER_NORM = VIEWPORT_H / SCALE / 2

# Below this descent rate, above this altitude, the lander is dawdling rather
# than landing. Normalised units, matching the observation.
HOVER_SPEED = 0.05
HOVER_MIN_ALTITUDE = 0.1

# Safe touchdown speed in normalised units (1.42 world u/s, measured Arc 2
# Phase 3). Main-engine frames fired while already slower than this buy
# nothing.
SAFE_TOUCHDOWN_NORM = 1.42 / VY_NORM_TO_WORLD


def segment_of(y: float, contact: bool) -> str:
    """Which segment a single step belongs to.

    Contact wins over altitude: once a leg is down the lander is settling
    whatever the altitude reading says.
    """
    if contact:
        return "SETTLING"
    if y >= BAND_EDGES[0]:
        return "DESCENT"
    if y >= BAND_EDGES[1]:
        return "APPROACH"
    return "TERMINAL"


def record_episode(
    controller: BaseController,
    seed: int,
    env_name: str = "LunarLander-v3",
    env_kwargs: Optional[Dict] = None,
    max_steps: int = 2000,
) -> pd.DataFrame:
    """Run one episode and return a per-step trace. The only env-touching part."""
    env = gym.make(env_name, **(env_kwargs or {}))
    observation, _ = env.reset(seed=seed)
    rows = []
    terminated = truncated = False
    steps = 0

    while not (terminated or truncated) and steps < max_steps:
        action = controller.get_action(observation)
        rows.append(
            {
                "y": float(observation[1]),
                "vy": float(observation[3]),
                "x": float(observation[0]),
                "vx": float(observation[2]),
                "angle": float(observation[4]),
                "omega": float(observation[5]),
                "contact": bool(observation[6] or observation[7]),
                "action": action,
            }
        )
        observation, reward, terminated, truncated, _ = env.step(action)
        rows[-1]["reward"] = float(reward)
        steps += 1

    env.close()
    return pd.DataFrame(rows)


def summarise_segments(steps: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Ticks, fuel and engine usage per segment. Segments with no steps are
    still reported, with zeros -- an absent segment is information."""
    labels = [segment_of(r.y, r.contact) for r in steps.itertuples()]
    out: Dict[str, Dict[str, float]] = {}

    for name in SEGMENTS:
        rows = steps[[label == name for label in labels]]
        main = side = 0
        for action in rows["action"]:
            m, s = count_engine_frames(action)
            main += m
            side += s
        out[name] = {
            "ticks": len(rows),
            "main_frames": main,
            "side_frames": side,
            "fuel": main * MAIN_ENGINE_REWARD_COST + side * SIDE_ENGINE_REWARD_COST,
        }
    return out


def behaviour_scores(steps: pd.DataFrame) -> Dict[str, float]:
    """Named behaviours, each measurable and each traceable to a finding."""
    if steps.empty:
        return {
            "hover_ticks": 0,
            "worst_tilt": 0.0,
            "attitude_reversals": 0,
            "wasted_main_frames": 0,
            "lateral_waste": 0.0,
        }

    airborne = ~steps["contact"].to_numpy(dtype=bool)
    y = steps["y"].to_numpy(dtype=float)
    vy = steps["vy"].to_numpy(dtype=float)

    # Dawdling: barely descending while there is still height to lose. The
    # altitude floor matters -- creeping down just above the pad is correct
    # terminal behaviour, not hovering.
    hover = airborne & (np.abs(vy) < HOVER_SPEED) & (y > HOVER_MIN_ALTITUDE)

    # A controller fighting itself: sign changes in angular velocity.
    omega = steps["omega"].to_numpy(dtype=float)
    signs = np.sign(omega[np.abs(omega) > 1e-9])
    reversals = int(np.count_nonzero(np.diff(signs))) if signs.size > 1 else 0

    # Thrust spent while already descending slower than a safe touchdown.
    main_fired = np.array(
        [count_engine_frames(a)[0] for a in steps["action"]], dtype=bool
    )
    wasted = main_fired & airborne & (np.abs(vy) < SAFE_TOUCHDOWN_NORM)

    # Lateral path travelled versus net displacement: drifting out and back
    # is motion that achieved nothing.
    x = steps["x"].to_numpy(dtype=float)
    path = float(np.abs(np.diff(x)).sum()) if x.size > 1 else 0.0
    net = abs(float(x[-1] - x[0]))
    lateral_waste = (path - net) / max(net, 1e-6) if path > net + 1e-12 else 0.0

    return {
        "hover_ticks": int(hover.sum()),
        "worst_tilt": float(np.abs(steps["angle"].to_numpy(dtype=float)).max()),
        "attitude_reversals": reversals,
        "wasted_main_frames": int(wasted.sum()),
        "lateral_waste": float(lateral_waste),
    }


def _mean_of_known(values: Sequence[float]) -> float:
    known = [v for v in values if not np.isnan(v)]
    return float(np.mean(known)) if known else float("nan")


def grade(actual_ticks: float, ideal_ticks: float) -> float:
    """`ideal / actual`, capped at 1.0.

    Capped because beating the idealized plan does not mean the controller is
    better than possible -- it means the model is wrong, which is a finding
    to investigate rather than a grade to award. Returns NaN when the ideal
    never enters the segment, since there is nothing to grade against.
    """
    if ideal_ticks <= 0:
        return float("nan")
    if actual_ticks <= 0:
        return 1.0
    return min(1.0, ideal_ticks / actual_ticks)


def ideal_segment_ticks(
    steps: pd.DataFrame,
    const: Optional[DescentConstants] = None,
    safe_touchdown_speed: float = 1.42,
) -> Dict[str, int]:
    """Run the idealized descent from this episode's own start state and
    segment it the same way.

    This is what makes the grades apples-to-apples: rather than comparing a
    segment against an arbitrary target, it is compared against how long the
    optimal plan spends in that same altitude band starting from the same
    place. The plan is 1-D (no lateral or rotational dynamics), so these are
    optimistic -- see Arc 3 Phase A.
    """
    const = const if const is not None else measure_descent_constants()
    if steps.empty:
        return {name: 0 for name in SEGMENTS}

    h0 = float(steps["y"].iloc[0]) * Y_WORLD_PER_NORM
    v0 = float(steps["vy"].iloc[0]) * VY_NORM_TO_WORLD

    plan = min_time_to_land(h0, v0, const, safe_touchdown_speed)
    switch = int(plan["switch_tick"])

    # Replay the plan, recording the band each tick falls in.
    counts = {name: 0 for name in SEGMENTS}
    h, v, t = h0, v0, 0
    while h > 0 and t < 5000:
        firing = t >= switch and (-v) > safe_touchdown_speed
        v += const.gravity * const.dt + (const.engine_dv_per_tick if firing else 0.0)
        h += v * const.dt
        counts[segment_of(h / Y_WORLD_PER_NORM, contact=False)] += 1
        t += 1
    return counts


@dataclass
class ReportCard:
    controller: str
    seed: int
    landed: bool
    total_reward: float
    segments: Dict[str, Dict[str, float]] = field(default_factory=dict)
    ideal_ticks: Dict[str, int] = field(default_factory=dict)
    grades: Dict[str, float] = field(default_factory=dict)
    behaviours: Dict[str, float] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "segment": name,
                    "ticks": self.segments[name]["ticks"],
                    "ideal": self.ideal_ticks.get(name, 0),
                    "grade": self.grades.get(name, float("nan")),
                    "fuel": self.segments[name]["fuel"],
                }
                for name in SEGMENTS
            ]
        )


def report_card(
    controller: BaseController,
    seed: int,
    name: str = "controller",
    const: Optional[DescentConstants] = None,
    env_kwargs: Optional[Dict] = None,
) -> ReportCard:
    """Full mark for one episode."""
    steps = record_episode(controller, seed, env_kwargs=env_kwargs)
    segments = summarise_segments(steps)
    ideal = ideal_segment_ticks(steps, const=const)
    return ReportCard(
        controller=name,
        seed=seed,
        landed=bool(steps["reward"].sum() >= 200) if not steps.empty else False,
        total_reward=float(steps["reward"].sum()) if not steps.empty else 0.0,
        segments=segments,
        ideal_ticks=ideal,
        grades={n: grade(segments[n]["ticks"], ideal.get(n, 0)) for n in SEGMENTS},
        behaviours=behaviour_scores(steps),
    )


def mark_controller(
    controller: BaseController,
    seeds: Sequence[int],
    name: str = "controller",
    const: Optional[DescentConstants] = None,
    env_kwargs: Optional[Dict] = None,
) -> pd.DataFrame:
    """Average report cards over several episodes into one marking frame."""
    const = const if const is not None else measure_descent_constants()
    cards = [report_card(controller, s, name, const, env_kwargs) for s in seeds]
    landed = [c for c in cards if c.landed]
    source = landed or cards

    rows = []
    for segment in SEGMENTS:
        rows.append(
            {
                "controller": name,
                "segment": segment,
                "ticks": float(np.mean([c.segments[segment]["ticks"] for c in source])),
                "ideal": float(np.mean([c.ideal_ticks.get(segment, 0) for c in source])),
                # SETTLING has no ideal to grade against -- the descent model
                # stops at ground contact -- so every card is NaN there and
                # nanmean would warn on an empty slice.
                "grade": _mean_of_known([c.grades[segment] for c in source]),
                "fuel": float(np.mean([c.segments[segment]["fuel"] for c in source])),
            }
        )
    frame = pd.DataFrame(rows)
    frame.attrs["behaviours"] = {
        key: float(np.mean([c.behaviours[key] for c in source]))
        for key in cards[0].behaviours
    }
    frame.attrs["success_rate_pct"] = 100.0 * len(landed) / len(cards)
    return frame
