# Lunar Lander Lab

A Python workspace for experimenting with classical (rule-based) control and
reinforcement learning on Gymnasium's `LunarLander-v3`.

## Project Structure

```
lunar_lander_lab/
├── README.md
├── requirements.txt
├── main.py                  # CLI entry point
├── controllers/
│   ├── __init__.py
│   ├── base.py               # BaseController abstract interface
│   ├── heuristic.py          # HeuristicController: rule-based PID-style logic
│   └── rl_agent.py           # RLAgent: PPO wrapper (Stable-Baselines3)
├── utils/
│   ├── __init__.py
│   └── evaluation.py         # run_benchmark(): head-to-head controller comparison
└── models/                   # Saved PPO checkpoints (.zip)
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

Requires Python 3.10-3.12 (Box2D and PyTorch wheels; newer interpreters may
lack prebuilt wheels for these packages).

## CLI Usage

Run from inside `lunar_lander_lab/`.

### Render a controller landing an episode

```bash
python main.py run --controller heuristic
python main.py run --controller rl
```

Opens a Pygame window and renders one episode. `--controller rl` requires a
trained model (see below).

### Train a PPO agent

```bash
python main.py train --timesteps 100000
```

Trains a PPO policy (Stable-Baselines3, `MlpPolicy`) on `LunarLander-v3` and
saves the checkpoint to `models/ppo_lunar_lander.zip`.

### Benchmark controllers side-by-side

```bash
python main.py benchmark --episodes 50
```

Runs the heuristic controller and (if trained) the PPO agent across the same
set of seeds, prints a metrics table (mean reward, success/landing rate,
average flight time, crash rate), and saves a comparison chart to
`benchmark_results.png`.

## Controllers

- **`HeuristicController`** — proportional control on horizontal position,
  velocity, angle and descent speed. Gains are class attributes
  (`ANGLE_GAIN_POS`, `DESCENT_GAIN`, etc.) for easy tuning.
- **`RLAgent`** — trains/loads a PPO model and selects actions via
  `model.predict(obs, deterministic=True)`.

Both implement `BaseController.get_action(observation) -> int`, so new
controllers can be dropped in and benchmarked the same way.
