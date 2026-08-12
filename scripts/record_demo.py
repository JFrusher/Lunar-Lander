"""Record a controller landing as a GIF.

Reproducible and scriptable — renders via `rgb_array` and encodes with
imageio, rather than screen-capturing a `render_mode="human"` window.

    python scripts/record_demo.py --controller heuristic --output docs/media/heuristic.gif
    python scripts/record_demo.py --controller rl --model <path-to.zip> --output docs/media/rl.gif
"""

import argparse
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio

from lunar_lander_lab.cli import build_controller
from lunar_lander_lab.utils.evaluation import SUCCESS_REWARD_THRESHOLD


def record(
    controller,
    output: Path,
    seed: int,
    fps: int = 50,
    frame_skip: int = 2,
    max_steps: int = 1000,
) -> dict:
    """Run one episode, writing every `frame_skip`-th frame to `output`."""
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    observation, _ = env.reset(seed=seed)
    # Planners carry a warm-started plan between steps; without this a retry
    # would begin seeded by the end of the previous episode.
    if hasattr(controller, "reset"):
        controller.reset()

    frames = []
    total_reward = 0.0
    steps = 0
    terminated = truncated = False

    while not (terminated or truncated) and steps < max_steps:
        if steps % frame_skip == 0:
            frames.append(env.render())
        action = controller.get_action(observation)
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        steps += 1

    # Hold the final frame so the landing is readable before the loop restarts.
    frames.extend([env.render()] * (fps // frame_skip))
    env.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output, frames, fps=fps // frame_skip, loop=0)

    return {
        "reward": total_reward,
        "steps": steps,
        "landed": total_reward >= SUCCESS_REWARD_THRESHOLD,
        "frames": len(frames),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controller", choices=["heuristic", "scheduled", "rl", "mpc"], required=True
    )
    parser.add_argument("--model", default=None, help="Checkpoint path/name for --controller rl")
    parser.add_argument(
        "--gains", default=None,
        help="Alternative gains file, for recording an older controller side by side",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument(
        "--require-landing", action="store_true",
        help="Retry seeds until the episode lands successfully (for demo GIFs)",
    )
    args = parser.parse_args()

    controller = build_controller(
        args.controller, model_name=args.model, gains_path=args.gains
    )

    seed = args.seed if args.seed is not None else 0
    while True:
        result = record(controller, args.output, seed=seed, frame_skip=args.frame_skip)
        print(f"seed {seed}: reward {result['reward']:.1f}, {result['steps']} steps, "
              f"landed={result['landed']}, {result['frames']} frames -> {args.output}")
        if result["landed"] or not args.require_landing or args.seed is not None:
            break
        seed += 1


if __name__ == "__main__":
    main()
