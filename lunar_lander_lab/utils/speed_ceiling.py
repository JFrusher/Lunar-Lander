"""Idealized point-mass minimum-time descent bound.

Ignores horizontal motion, rotation, and per-tick thrust-impulse noise --
this is a lower bound on landing time, not a solve of the real Box2D
dynamics. See tmp/SPEED_ROADMAP.md, Phase 1.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd
from gymnasium.envs.box2d.lunar_lander import FPS, LEG_DOWN, SCALE

from .paths import new_run_dir

# Chart tokens shared with time_penalty.py so every figure in this repo reads
# as one system (validated CVD-safe pair -- see that module's note).
from .time_penalty import _CONTROLLER_COLORS, _GRID, _INK_MUTED, _INK_PRIMARY, _SURFACE

DT = 1.0 / FPS

# The shipped heuristic controller's own TARGET_DESCENT_SPEED gain (-0.0736,
# in the env's normalized units) converted to raw world units/s via the
# normalization factor documented in lunar_lander.py's docstring (vy scale
# 7.5, i.e. divide by FPS/(VIEWPORT_H/SCALE/2) -- equivalently multiply the
# normalized value by (VIEWPORT_H/SCALE/2)/FPS = 0.1333). Not an arbitrary
# round number: it's what the tuned, 98%-successful controller already
# targets as a safe controlled-descent speed.
SHIPPED_HEURISTIC_TARGET_SPEED = 0.0736 / (1 / 7.5)

# Held-out landing steps from tmp/ROADMAP.md's Phase 4 (multi-seed sweep),
# for expressing this phase's ceiling as "% of current best".
CURRENT_BEST_STEPS = {
    "RL (PPO), penalty=0.1": 250.5,
    "Heuristic, penalty=0.0": 400.8,
}

# The ceiling bars get a recessive neutral rather than a third categorical
# hue: they're the reference the controllers are measured against, not a
# competing series. Controllers keep their usual identity colors.
_CEILING_INK = "#b9b7b0"


@dataclass
class DescentConstants:
    """Physical constants the descent model needs.

    `mass` isn't used by the simulation below -- `engine_dv_per_tick` is
    measured empirically as a velocity change, which already has mass's
    effect baked in. It's kept here purely as a reported/documented value,
    since the roadmap phase this module implements explicitly asks for it.
    """

    mass: float
    gravity: float  # negative, world units/s^2
    engine_dv_per_tick: float  # measured mean upward delta-v from one tick of main-engine fire
    dt: float


def measure_descent_constants(n_samples: int = 20, seed_start: int = 0) -> DescentConstants:
    """Empirically measure the constants the descent model needs.

    Mass and the main engine's per-tick impulse aren't plain module
    constants (mass comes from Box2D's fixture-density integration, the
    impulse depends on the lander's orientation and a small random
    dispersion) -- reading them off a live env avoids re-deriving that
    geometry by hand.
    """
    env = gym.make("LunarLander-v3")
    env.reset(seed=seed_start)
    mass = env.unwrapped.lander.mass
    gravity = env.unwrapped.gravity

    dvs = []
    for s in range(seed_start, seed_start + n_samples):
        env.reset(seed=s)
        vy_before = env.unwrapped.lander.linearVelocity.y
        env.step(2)
        vy_after = env.unwrapped.lander.linearVelocity.y
        dvs.append((vy_after - vy_before) - gravity * DT)
    env.close()

    return DescentConstants(
        mass=mass,
        gravity=gravity,
        engine_dv_per_tick=sum(dvs) / len(dvs),
        dt=DT,
    )


def sample_initial_states(n_samples: int = 20, seed_start: int = 0) -> List[Tuple[float, float]]:
    """Empirically sample (altitude above touchdown, vertical velocity) at the
    first observation a controller actually sees.

    `reset()` runs one physics tick internally before returning, so this
    samples the real starting condition a controller acts on -- not the
    pre-tick spawn point.
    """
    env = gym.make("LunarLander-v3")
    states = []
    for s in range(seed_start, seed_start + n_samples):
        env.reset(seed=s)
        u = env.unwrapped
        touchdown_y = u.helipad_y + LEG_DOWN / SCALE
        states.append((u.lander.position.y - touchdown_y, u.lander.linearVelocity.y))
    env.close()
    return states


def simulate_descent(
    switch_tick: int,
    h0: float,
    v0: float,
    const: DescentConstants,
    safe_touchdown_speed: float,
    max_ticks: int = 5000,
) -> Tuple[int, float]:
    """Simulate one bang-bang trajectory: free-fall (coast) unconditionally
    for `switch_tick` ticks, then -- since the main engine measures stronger
    than gravity (net delta-v while firing is positive) -- fire only on the
    ticks after that where current descent speed still exceeds
    `safe_touchdown_speed`, coasting otherwise. That threshold-following rule
    is self-limiting (it stops firing the instant it's no longer needed), so
    unlike firing unconditionally after the switch, it can't overshoot into
    sustained ascent -- braking only ever engages long enough to hold speed
    near the cap.

    Returns (ticks_to_touchdown, touchdown_speed) where touchdown_speed is
    positive-down (Box2D's vertical velocity is positive-up).
    """
    h, v, t = h0, v0, 0
    while h > 0 and t < max_ticks:
        firing = t >= switch_tick and (-v) > safe_touchdown_speed
        v += const.gravity * const.dt + (const.engine_dv_per_tick if firing else 0.0)
        h += v * const.dt
        t += 1
    return t, -v


def min_time_to_land(
    h0: float,
    v0: float,
    const: DescentConstants,
    safe_touchdown_speed: float,
    max_ticks: int = 5000,
) -> dict:
    """Find the latest (most free-fall, hence fastest) switch point that
    still touches down within `safe_touchdown_speed`.

    A linear scan from the latest switch_tick downward, not a binary search:
    switching later leaves less remaining altitude for the threshold-follower
    to bring speed back under the cap before touchdown, but exactly how
    little is safe isn't assumed to be a clean single threshold -- a plain
    scan is correct regardless of that shape, and the candidate count here
    (bounded by the free-fall-only landing time) is small enough that it
    costs nothing.
    """
    free_fall_ticks, _ = simulate_descent(
        switch_tick=max_ticks, h0=h0, v0=v0, const=const,
        safe_touchdown_speed=safe_touchdown_speed, max_ticks=max_ticks,
    )

    for switch_tick in range(free_fall_ticks, -1, -1):
        ticks, speed = simulate_descent(
            switch_tick, h0, v0, const, safe_touchdown_speed, max_ticks=max_ticks
        )
        landed = ticks < max_ticks
        if landed and speed <= safe_touchdown_speed:
            return {
                "feasible": True,
                "switch_tick": switch_tick,
                "ticks": ticks,
                "touchdown_speed": speed,
            }

    # Even threshold-following for the entire flight doesn't land safely
    # within max_ticks -- report that gentlest attempt as the closest failure.
    ticks, speed = simulate_descent(0, h0, v0, const, safe_touchdown_speed, max_ticks=max_ticks)
    return {"feasible": False, "switch_tick": 0, "ticks": ticks, "touchdown_speed": speed}


def ceiling_chart_data(
    rows: List[dict], measured: Optional[Dict[str, float]] = None
) -> List[Tuple[str, float, bool]]:
    """Merge idealized ceiling rows with measured controller results into a
    single ascending list of `(label, ticks, is_ceiling)`.

    Sorted fastest-first so the chart reads as one continuous scale from
    "physically possible" up to "what we actually do" -- the gap between
    those is the whole point of Phase 1.
    """
    measured = CURRENT_BEST_STEPS if measured is None else measured
    entries = [
        (f"ceiling @ {r['safe_touchdown_speed']:.2f} u/s", r["mean_ticks"], True) for r in rows
    ]
    entries += [(label, ticks, False) for label, ticks in measured.items()]
    return sorted(entries, key=lambda e: e[1])


def _plot_ceiling(entries: List[Tuple[str, float, bool]], path: Path) -> None:
    """Horizontal bars: idealized floors recessive grey, real controllers in
    their usual identity colors (blue heuristic / orange RL, as everywhere
    else in this repo). The ceiling isn't an entity competing with them --
    it's the reference they're measured against, so it stays recessive."""
    labels = [e[0] for e in entries]
    values = [e[1] for e in entries]
    colors = [
        _CEILING_INK if is_ceiling
        else (_CONTROLLER_COLORS["Heuristic"] if "Heuristic" in label
              else _CONTROLLER_COLORS["RL (PPO)"])
        for label, _, is_ceiling in entries
    ]

    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=_SURFACE)
    ax.set_facecolor(_SURFACE)
    positions = range(len(entries))
    ax.barh(list(positions), values, color=colors, height=0.62)

    for pos, value in zip(positions, values):
        ax.text(value + max(values) * 0.012, pos, f"{value:.0f}",
                va="center", fontsize=9, color=_INK_PRIMARY)

    ax.set_yticks(list(positions), labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("ticks to touchdown (lower is faster)", color=_INK_PRIMARY)
    ax.set_xlim(0, max(values) * 1.12)
    ax.grid(True, axis="x", color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(_GRID)
    ax.tick_params(colors=_INK_MUTED)

    ax.set_title(
        "Idealized minimum-time descent vs. measured controllers",
        color=_INK_PRIMARY, fontsize=12, pad=12,
    )
    fig.text(0.5, -0.04,
             "Grey = point-mass bang-bang lower bound (no horizontal or rotational dynamics). "
             "Colored = measured, held-out episodes.",
             ha="center", fontsize=8, color=_INK_MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=_SURFACE, bbox_inches="tight")
    plt.close(fig)


def run_speed_ceiling(
    safe_touchdown_speeds: Tuple[float, ...] = (
        SHIPPED_HEURISTIC_TARGET_SPEED,
        0.3,
        1.0,
    ),
    n_samples: int = 20,
    seed_start: int = 0,
    output_dir: Optional[str] = None,
) -> List[dict]:
    """Print and return the idealized minimum-time ceiling at a few candidate
    safe-touchdown-speed assumptions, alongside the current shipped best.
    Writes a CSV + chart to `runs/speed_ceiling/<timestamp>/`.

    Solved per sampled (h0, v0) start state and averaged over the *results*,
    not solved once on the average start state -- v0 in particular varies in
    sign across resets (the initial random push can go either way), and
    averaging inputs first would hand the solver a start state no real
    episode ever has.
    """
    const = measure_descent_constants(n_samples=n_samples, seed_start=seed_start)
    states = sample_initial_states(n_samples=n_samples, seed_start=seed_start)

    print(
        f"Measured: mass={const.mass:.3f} gravity={const.gravity:.2f} "
        f"engine_dv/tick={const.engine_dv_per_tick:.4f} dt={const.dt:.4f}"
    )
    print(f"Start states: {n_samples} resets, h0 in "
          f"[{min(s[0] for s in states):.2f}, {max(s[0] for s in states):.2f}], "
          f"v0 in [{min(s[1] for s in states):.2f}, {max(s[1] for s in states):.2f}]")

    rows = []
    for speed in safe_touchdown_speeds:
        results = [min_time_to_land(h0, v0, const, speed) for h0, v0 in states]
        infeasible = sum(not r["feasible"] for r in results)
        ticks = [r["ticks"] for r in results]
        mean_ticks = sum(ticks) / len(ticks)

        comparisons = ", ".join(
            f"{100 * mean_ticks / best:.0f}% of {label}" for label, best in CURRENT_BEST_STEPS.items()
        )
        print(
            f"  safe touchdown <= {speed:.3f} u/s: {mean_ticks:.1f} ticks avg "
            f"(range {min(ticks)}-{max(ticks)}, {infeasible}/{len(states)} infeasible) "
            f"({comparisons})"
        )
        rows.append(
            {
                "safe_touchdown_speed": speed,
                "mean_ticks": mean_ticks,
                "min_ticks": min(ticks),
                "max_ticks": max(ticks),
                "infeasible_count": infeasible,
            }
        )

    out_dir = Path(output_dir) if output_dir else new_run_dir("speed_ceiling")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "speed_ceiling.csv", index=False)
    plot_path = out_dir / "speed_ceiling.png"
    _plot_ceiling(ceiling_chart_data(rows), plot_path)
    print(f"\nCSV:  {out_dir / 'speed_ceiling.csv'}")
    print(f"Plot: {plot_path}")
    return rows
