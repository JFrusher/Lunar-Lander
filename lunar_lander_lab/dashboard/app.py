"""Streamlit dashboard: pick a controller, watch it land, compare it against
others, or kick off training -- all reusing cli.py's/utils'/registry's
existing machinery. No new scoring or training logic lives here.

    streamlit run lunar_lander_lab/dashboard/app.py

Absolute imports throughout (not relative): Streamlit executes this file
directly rather than as part of the `lunar_lander_lab` package, matching the
same convention `scripts/record_demo.py` already uses for the same reason.
"""

import subprocess
import sys
import time

import gymnasium as gym
import streamlit as st

from lunar_lander_lab.controllers import build_controller, controller_names
from lunar_lander_lab.controllers.rl_agent import ALGOS, DEFAULT_MODEL_NAMES
from lunar_lander_lab.utils.evaluation import evaluate_controller_natural
from lunar_lander_lab.utils.marking import mark_controller
from lunar_lander_lab.utils.speed_ceiling import measure_descent_constants

st.set_page_config(page_title="Lunar Lander Lab", layout="wide")
st.title("Lunar Lander Lab")


def _try_build_controller(name: str):
    """(controller, None) on success, (None, message) on a missing checkpoint
    -- the common first-run case (nothing trained yet) shouldn't crash the
    page with a raw FileNotFoundError traceback."""
    try:
        return build_controller(name), None
    except FileNotFoundError:
        return None, f'No trained model found for "{name}" -- train one in the Train tab first.'


def _play_episode(controller, seed: int, frame_skip: int = 2, max_steps: int = 1000):
    """Run one episode, redrawing a placeholder with each captured frame.
    This *is* the dashboard's visualization -- no separate render pipeline."""
    env = gym.make("LunarLander-v3", render_mode="rgb_array", **controller.env_kwargs)
    if hasattr(controller, "reset"):
        controller.reset()
    observation, _ = env.reset(seed=seed)

    frame_slot = st.empty()
    total_reward = 0.0
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated) and steps < max_steps:
        if steps % frame_skip == 0:
            frame_slot.image(env.render())
            time.sleep(frame_skip / 50)
        action = controller.get_action(observation)
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        steps += 1
    frame_slot.image(env.render())
    env.close()
    return total_reward, steps


tab_run, tab_compare, tab_train = st.tabs(["Run & Visualize", "Compare", "Train"])

with tab_run:
    st.subheader("Run one episode")
    name = st.selectbox("Controller", controller_names(), key="run_controller")
    seed = st.number_input("Seed", min_value=0, value=0, key="run_seed")
    if st.button("Fly it"):
        controller, error = _try_build_controller(name)
        if error:
            st.error(error)
        else:
            with st.spinner(f"Flying {name}..."):
                reward, steps = _play_episode(controller, seed=int(seed))
            landed = reward >= 200
            (st.success if landed else st.warning)(
                f"reward={reward:.1f}  steps={steps}  landed={landed}"
            )

with tab_compare:
    st.subheader("Compare controllers")
    st.caption("Aggregate metrics (utils/evaluation.py), then per-segment report cards (utils/marking.py).")
    names = st.multiselect(
        "Controllers", controller_names(), default=["heuristic", "scheduled"], key="compare_controllers"
    )
    episodes = st.slider("Episodes per controller", 5, 100, 30, key="compare_episodes")

    if st.button("Run comparison") and names:
        rows = []
        built = {}
        with st.spinner("Evaluating..."):
            for cname in names:
                controller, error = _try_build_controller(cname)
                if error:
                    st.warning(error)
                    continue
                built[cname] = controller
                metrics = evaluate_controller_natural(
                    controller, num_episodes=int(episodes), env_kwargs=controller.env_kwargs
                )
                rows.append({"controller": cname, **metrics})

        if rows:
            st.dataframe(rows, use_container_width=True)

            st.subheader("Segment report cards")
            const = measure_descent_constants()
            seeds = list(range(50_000, 50_000 + int(episodes)))
            for cname, controller in built.items():
                frame = mark_controller(
                    controller, seeds, name=cname, const=const, env_kwargs=controller.env_kwargs
                )
                with st.expander(f"{cname}  (success {frame.attrs['success_rate_pct']:.0f}%)"):
                    st.dataframe(frame.drop(columns=["controller"]), use_container_width=True)

with tab_train:
    st.subheader("Train an RL agent")
    st.caption(
        "Shells out to the real `lunar-lander train` CLI and tails its output -- "
        "the checkpoint lands in runs/train/ and is picked up automatically by "
        "the Run/Compare tabs (they always resolve to the most recent run)."
    )
    algo = st.selectbox("Algorithm", sorted(ALGOS), key="train_algo")
    timesteps = st.number_input(
        "Timesteps", min_value=100, value=100_000, step=1_000, key="train_timesteps"
    )
    if st.button("Start training"):
        cmd = [
            sys.executable, "-m", "lunar_lander_lab.cli", "train",
            "--algo", algo, "--timesteps", str(int(timesteps)),
        ]
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        log_slot = st.empty()
        lines: list = []
        with st.spinner(f"Training {algo}..."):
            for line in process.stdout:
                lines.append(line.rstrip("\n"))
                log_slot.code("\n".join(lines[-40:]))
        returncode = process.wait()
        if returncode == 0:
            st.success(f'Done. Checkpoint: "{DEFAULT_MODEL_NAMES[algo]}" (runs/train/<latest>/).')
        else:
            st.error(f"Training exited with code {returncode}")
