# Lunar Lander Lab

A Python workspace for experimenting with classical (rule-based) control and
reinforcement learning on Gymnasium's `LunarLander-v3`.

## Project Structure

```
Lunar/
├── README.md
├── pyproject.toml
├── runs/                      # gitignored — all run output, kept forever
│   ├── train/<timestamp>/     # PPO checkpoints (.zip)
│   ├── pid_search/<timestamp>/ # gain-sweep datasets + plots
│   └── benchmark/<timestamp>/ # comparison charts
└── lunar_lander_lab/
    ├── cli.py                 # CLI entry point
    ├── configs/
    │   └── heuristic_gains.json   # tuned heuristic gains, hand-updated from sweeps
    ├── controllers/
    │   ├── base.py             # BaseController abstract interface
    │   ├── heuristic.py         # HeuristicController: rule-based PID-style logic
    │   └── rl_agent.py          # RLAgent: PPO wrapper (Stable-Baselines3)
    └── utils/
        ├── paths.py             # runs/ directory helpers
        ├── evaluation.py        # run_benchmark(): head-to-head controller comparison
        └── pid_search.py        # Monte Carlo gain sweep
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux
pip install -e .
```

Requires Python 3.10-3.12 (Box2D and PyTorch wheels; newer interpreters may
lack prebuilt wheels for these packages).

## CLI Usage

Runnable from anywhere once installed, as `lunar-lander`, or from the repo
root as `python -m lunar_lander_lab.cli`.

### Render a controller landing an episode

```bash
lunar-lander run --controller heuristic
lunar-lander run --controller rl
```

Opens a Pygame window and renders one episode. `--controller rl` loads the
most recently trained checkpoint under `runs/train/` (or pass an explicit
path with `--model`).

### Train a PPO agent

```bash
lunar-lander train --timesteps 100000
```

Trains a PPO policy (Stable-Baselines3, `MlpPolicy`) on `LunarLander-v3` and
saves the checkpoint to `runs/train/<timestamp>/ppo_lunar_lander.zip`.

### Benchmark controllers side-by-side

```bash
lunar-lander benchmark --episodes 50
```

Runs the heuristic controller and (if trained) the PPO agent across the same
set of seeds, prints a metrics table (mean reward, success/landing rate,
average flight time, crash rate), and saves a comparison chart to
`runs/benchmark/<timestamp>/benchmark_results.png`.

### Sweep heuristic gains

```bash
lunar-lander pid-search --samples 200 --episodes 30
```

Latin-Hypercube-samples the gain space, evaluates each set across seeded
episodes (multiprocessed), and writes a dataset + plots to
`runs/pid_search/<timestamp>/`.

To apply a winning sweep: open `runs/pid_search/<timestamp>/best_gains.json`
and hand-copy its `best_gains` values into
`lunar_lander_lab/configs/heuristic_gains.json`. This is a deliberate manual
step — a smaller or noisier sweep shouldn't be able to silently regress the
tuned default.

## Controllers

- **`HeuristicController`** — proportional control on horizontal position,
  velocity, angle and descent speed. Swept gains load from
  `configs/heuristic_gains.json`; unswept gains are class attributes.
- **`RLAgent`** — trains/loads a PPO model and selects actions via
  `model.predict(obs, deterministic=True)`.

Both implement `BaseController.get_action(observation) -> int`, so new
controllers can be dropped in and benchmarked the same way.
