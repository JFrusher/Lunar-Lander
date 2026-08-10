# Repo Reorg Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `lunar_lander_lab` into an installable package, unify all run output under one gitignored `runs/` directory, and move heuristic gains out of source into a config file — with zero change to controller logic, CLI commands, or algorithms.

**Architecture:** `lunar_lander_lab/` becomes a real Python package (`__init__.py`, relative imports) with a `pyproject.toml` + console entry point at repo root. A new `utils/paths.py` gives train/pid-search/benchmark one shared timestamped-directory convention (`runs/<kind>/<timestamp>/`) instead of three ad-hoc ones. Heuristic gains move from hardcoded class attributes to `configs/heuristic_gains.json`, read at import time.

**Tech Stack:** Python 3.10-3.12, setuptools (pyproject.toml build backend), no new runtime dependencies.

## Global Constraints

- No behavior change to controllers, CLI commands, or algorithms (per spec).
- `runs/` kept forever, gitignored wholesale — no auto-pruning.
- `configs/heuristic_gains.json` always exists (committed) — no fallback-if-missing branch in `heuristic.py`.
- Gains promotion (sweep result → config) stays manual — no auto-write from pid-search into `configs/`.
- Repo stays single-project — no `src/` layout, no multi-package scaffolding.

Full design context: `docs/superpowers/specs/2026-08-10-repo-reorg-design.md`

---

### Task 1: Package skeleton — `__init__.py`, `cli.py`, relative imports

**Files:**
- Create: `lunar_lander_lab/__init__.py`
- Create: `lunar_lander_lab/cli.py` (moved from `lunar_lander_lab/main.py`, content unchanged except imports)
- Delete: `lunar_lander_lab/main.py`
- Modify: `lunar_lander_lab/utils/evaluation.py:9`
- Modify: `lunar_lander_lab/utils/pid_search.py:18,66`

**Interfaces:**
- Produces: `lunar_lander_lab` importable as a package; `lunar_lander_lab.cli:main` callable (used by Task 2's console script).

- [ ] **Step 1: Create empty package marker**

Create `lunar_lander_lab/__init__.py` as a zero-byte empty file (no content — just makes the directory an importable package).

- [ ] **Step 2: Move main.py to cli.py with relative imports**

```bash
git mv lunar_lander_lab/main.py lunar_lander_lab/cli.py
```

Edit `lunar_lander_lab/cli.py` — change only the two import lines near the top:

```python
from controllers import HeuristicController, RLAgent
from utils.evaluation import SUCCESS_REWARD_THRESHOLD, run_benchmark
from utils.pid_search import run_monte_carlo
```
becomes:
```python
from .controllers import HeuristicController, RLAgent
from .utils.evaluation import SUCCESS_REWARD_THRESHOLD, run_benchmark
from .utils.pid_search import run_monte_carlo
```

Everything else in the file (all `cmd_*` functions, `build_controller`, `main()`, argparse setup) is unchanged.

- [ ] **Step 3: Fix evaluation.py's absolute import**

In `lunar_lander_lab/utils/evaluation.py`, line 9:
```python
from controllers.base import BaseController
```
becomes:
```python
from ..controllers.base import BaseController
```

- [ ] **Step 4: Fix pid_search.py's absolute imports**

In `lunar_lander_lab/utils/pid_search.py`, line 18:
```python
from utils.evaluation import CRASH_REWARD_THRESHOLD, SUCCESS_REWARD_THRESHOLD
```
becomes:
```python
from .evaluation import CRASH_REWARD_THRESHOLD, SUCCESS_REWARD_THRESHOLD
```

And inside `_evaluate_gain_set` (around line 66):
```python
    from controllers.heuristic import HeuristicController
```
becomes:
```python
    from ..controllers.heuristic import HeuristicController
```

- [ ] **Step 5: Verify the package imports cleanly**

Run from repo root (`/c/Projects/Lunar`):
```bash
python -m lunar_lander_lab.cli --help
```
Expected: argparse help text listing `run`, `train`, `benchmark`, `pid-search` subcommands. No `ImportError`/`ModuleNotFoundError`.

- [ ] **Step 6: Commit**

```bash
git add lunar_lander_lab/__init__.py lunar_lander_lab/cli.py lunar_lander_lab/main.py lunar_lander_lab/utils/evaluation.py lunar_lander_lab/utils/pid_search.py
git commit -m "Convert lunar_lander_lab into an importable package"
```

---

### Task 2: Packaging — `pyproject.toml`, drop `requirements.txt`, move README, update `.gitignore`

**Files:**
- Create: `pyproject.toml` (repo root)
- Delete: `lunar_lander_lab/requirements.txt`
- Move: `lunar_lander_lab/README.md` → `README.md` (repo root; content updated in Task 9)
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `lunar_lander_lab.cli:main` (from Task 1).
- Produces: `lunar-lander` console command once installed.

- [ ] **Step 1: Write pyproject.toml**

```toml
# pyproject.toml
[project]
name = "lunar-lander-lab"
version = "0.1.0"
description = "Classical control and RL experiments on Gymnasium's LunarLander-v3."
requires-python = ">=3.10,<3.13"
dependencies = [
    "gymnasium[box2d]>=1.0.0",
    "stable-baselines3>=2.3.0",
    "torch>=2.2.0",
    "matplotlib>=3.8.0",
    "numpy>=1.26.0",
    "pandas>=2.2.0",
]

[project.scripts]
lunar-lander = "lunar_lander_lab.cli:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["lunar_lander_lab*"]

[tool.setuptools.package-data]
lunar_lander_lab = ["configs/*.json"]
```

- [ ] **Step 2: Remove requirements.txt**

```bash
git rm lunar_lander_lab/requirements.txt
```

- [ ] **Step 3: Move README to repo root**

```bash
git mv lunar_lander_lab/README.md README.md
```
(Content rewritten in Task 9, after runs/configs mechanics exist to document.)

- [ ] **Step 4: Update .gitignore**

Current `.gitignore`:
```
.venv/
.claude/
**/tmp/
**/__pycache__/
lunar_lander_lab/models
lunar_lander_lab/pid_search_results_*
```
becomes:
```
.venv/
.claude/
**/tmp/
**/__pycache__/
runs/
*.egg-info/
build/
dist/
```

- [ ] **Step 5: Install and verify the console script**

```bash
pip install -e .
lunar-lander --help
```
Expected: same argparse help as Task 1's `python -m` check, now runnable from any directory.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml README.md .gitignore
git commit -m "Add pyproject.toml packaging, drop requirements.txt"
```

---

### Task 3: `utils/paths.py` — shared run-directory helper

**Files:**
- Create: `lunar_lander_lab/utils/paths.py`

**Interfaces:**
- Produces:
  - `RUNS_DIR: Path` — `<repo_root>/runs`
  - `new_run_dir(kind: str, base: Path = RUNS_DIR) -> Path` — creates and returns `base/kind/<YYYYMMDD_HHMMSS>/`
  - `latest_run_file(kind: str, filename: str, base: Path = RUNS_DIR) -> Path` — most recent `base/kind/*/filename`; raises `FileNotFoundError` if none exist
- Consumed by: Task 4 (`rl_agent.py`), Task 6 (`evaluation.py`), Task 7 (`pid_search.py`).

- [ ] **Step 1: Write paths.py with its self-check**

```python
# lunar_lander_lab/utils/paths.py
"""Shared paths: package root, runs directory, timestamped run dirs."""

from datetime import datetime
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PACKAGE_ROOT.parent
RUNS_DIR = REPO_ROOT / "runs"


def new_run_dir(kind: str, base: Path = RUNS_DIR) -> Path:
    """Create and return base/kind/<timestamp>/."""
    path = base / kind / datetime.now().strftime("%Y%m%d_%H%M%S")
    path.mkdir(parents=True, exist_ok=True)
    return path


def latest_run_file(kind: str, filename: str, base: Path = RUNS_DIR) -> Path:
    """Most recent base/kind/*/filename, by timestamp dir name."""
    matches = sorted((base / kind).glob(f"*/{filename}"))
    if not matches:
        raise FileNotFoundError(f"No {filename} found under {base / kind}")
    return matches[-1]


if __name__ == "__main__":
    import shutil
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    try:
        run_dir = new_run_dir("demo", base=tmp)
        assert run_dir.is_dir(), run_dir
        (run_dir / "result.txt").write_text("hello")

        found = latest_run_file("demo", "result.txt", base=tmp)
        assert found == run_dir / "result.txt", found
        assert found.read_text() == "hello"

        try:
            latest_run_file("missing", "x.txt", base=tmp)
            raise AssertionError("expected FileNotFoundError")
        except FileNotFoundError:
            pass

        print("paths self-check OK")
    finally:
        shutil.rmtree(tmp)
```

- [ ] **Step 2: Run the self-check**

```bash
python -m lunar_lander_lab.utils.paths
```
Expected: `paths self-check OK`, no assertion errors.

- [ ] **Step 3: Commit**

```bash
git add lunar_lander_lab/utils/paths.py
git commit -m "Add shared runs/ directory helper with self-check"
```

---

### Task 4: `rl_agent.py` — train/load via `runs/train/`

**Files:**
- Modify: `lunar_lander_lab/controllers/rl_agent.py`

**Interfaces:**
- Consumes: `new_run_dir("train")`, `latest_run_file("train", filename)` from Task 3.
- Produces: `RLAgent.train(...) -> str` (unchanged signature/return-type: full path to saved `.zip`), `RLAgent.load(model_path: str) -> None` (unchanged signature).

- [ ] **Step 1: Replace MODELS_DIR with paths helper, update train() and load()**

Remove:
```python
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
```

Add near the top imports:
```python
from ..utils.paths import latest_run_file, new_run_dir
```

Replace `train()`:
```python
    def train(
        self,
        env_name: str = "LunarLander-v3",
        total_timesteps: int = 100_000,
        save_path: str = "ppo_lunar_lander",
        hyperparams: Optional[dict] = None,
    ) -> str:
        """Train a PPO model from scratch and save the weights to runs/train/<timestamp>/."""
        params = {**DEFAULT_HYPERPARAMS, **(hyperparams or {})}
        env = DummyVecEnv([lambda: gym.make(env_name)])

        self.model = PPO(env=env, **params)
        self.model.learn(total_timesteps=total_timesteps)

        run_dir = new_run_dir("train")
        full_path = run_dir / save_path
        self.model.save(str(full_path))
        env.close()
        return str(full_path.with_suffix(".zip"))
```

Replace `load()`:
```python
    def load(self, model_path: str) -> None:
        """Load a previously trained PPO model.

        `model_path` may be an existing file path, or a checkpoint name
        (e.g. "ppo_lunar_lander") to resolve against the most recent
        runs/train/*/<name>.zip.
        """
        path = Path(model_path)
        if not path.exists():
            path = latest_run_file("train", f"{model_path}.zip")
        self.model = PPO.load(str(path))
```

`get_action()` is unchanged.

- [ ] **Step 2: Verify training writes into runs/train/**

```bash
python -m lunar_lander_lab.cli train --timesteps 2000
```
Expected: prints `Model saved to <repo_root>\runs\train\<timestamp>\ppo_lunar_lander.zip`; that file exists.

- [ ] **Step 3: Verify load resolves the checkpoint automatically**

```bash
python -m lunar_lander_lab.cli run --controller rl --seed 0
```
Expected: opens the Pygame window and renders an episode using the checkpoint just trained (no `FileNotFoundError`). Close the window / Ctrl+C to stop.

- [ ] **Step 4: Commit**

```bash
git add lunar_lander_lab/controllers/rl_agent.py
git commit -m "Save/load PPO checkpoints via runs/train/"
```

---

### Task 5: `heuristic.py` — gains from `configs/heuristic_gains.json`

**Files:**
- Create: `lunar_lander_lab/configs/heuristic_gains.json`
- Modify: `lunar_lander_lab/controllers/heuristic.py`

**Interfaces:**
- Produces: `HeuristicController` class attributes unchanged in name/type (still floats, still overridable via `setattr` — required by `pid_search.py`'s `_evaluate_gain_set`), now sourced from JSON instead of literals.

- [ ] **Step 1: Write the config file with the current (v2 sweep) best gains**

```json
{
  "ANGLE_GAIN_VEL": 1.175571804391529,
  "DESCENT_GAIN": 1.3919415369112689,
  "TARGET_DESCENT_SPEED": -0.10534496960720757,
  "ANGLE_THRESHOLD": 0.03654782964522024,
  "HOVER_THRESHOLD": 0.12111246024626458
}
```
Save as `lunar_lander_lab/configs/heuristic_gains.json`.

- [ ] **Step 2: Load gains at import time in heuristic.py**

Replace the whole file:
```python
"""Rule-based PID-style controller for LunarLander-v3."""

import json
from pathlib import Path
from typing import Sequence

from .base import BaseController

_GAINS_PATH = Path(__file__).resolve().parent.parent / "configs" / "heuristic_gains.json"
_GAINS = json.loads(_GAINS_PATH.read_text())


class HeuristicController(BaseController):
    """Hand-tuned proportional controller.

    Drives a target horizontal angle from horizontal position/velocity error,
    then converts angle/altitude error into engine firings. Gains are exposed
    as class attributes so they can be tweaked without touching the logic.
    """

    # Swept gains: configs/heuristic_gains.json (see utils/pid_search.py to re-tune)
    ANGLE_GAIN_VEL = _GAINS["ANGLE_GAIN_VEL"]
    DESCENT_GAIN = _GAINS["DESCENT_GAIN"]
    TARGET_DESCENT_SPEED = _GAINS["TARGET_DESCENT_SPEED"]
    ANGLE_THRESHOLD = _GAINS["ANGLE_THRESHOLD"]
    HOVER_THRESHOLD = _GAINS["HOVER_THRESHOLD"]

    # Never swept, held at original defaults:
    ANGLE_GAIN_POS = 0.5  # horizontal position -> desired angle
    ANGLE_ERROR_GAIN = 0.5  # angle error -> angular correction
    ANGULAR_VEL_GAIN = 1.0  # angular velocity damping

    def get_action(self, observation: Sequence[float]) -> int:
        x, y, vx, vy, angle, angular_vel, leg1, leg2 = observation

        if leg1 and leg2:
            return 0

        target_angle = self.ANGLE_GAIN_POS * x + self.ANGLE_GAIN_VEL * vx
        target_angle = max(-0.4, min(0.4, target_angle))

        angle_error = target_angle - angle
        angular_correction = (
            self.ANGLE_ERROR_GAIN * angle_error - self.ANGULAR_VEL_GAIN * angular_vel
        )

        hover_error = self.TARGET_DESCENT_SPEED - vy
        vertical_correction = self.DESCENT_GAIN * hover_error

        if angular_correction < -self.ANGLE_THRESHOLD:
            return 3  # fire right engine, rotate left
        if angular_correction > self.ANGLE_THRESHOLD:
            return 1  # fire left engine, rotate right
        if vertical_correction > self.HOVER_THRESHOLD:
            return 2  # fire main engine
        return 0
```

- [ ] **Step 3: Verify gains load and behavior is unchanged**

```bash
python -c "from lunar_lander_lab.controllers.heuristic import HeuristicController; c = HeuristicController(); print(c.DESCENT_GAIN, c.ANGLE_THRESHOLD)"
```
Expected: `1.3919415369112689 0.03654782964522024` (matches the JSON, matches the old hardcoded values — no behavior change).

- [ ] **Step 4: Commit**

```bash
git add lunar_lander_lab/configs/heuristic_gains.json lunar_lander_lab/controllers/heuristic.py
git commit -m "Load heuristic gains from configs/heuristic_gains.json"
```

---

### Task 6: `evaluation.py` — benchmark output via `runs/benchmark/`

**Files:**
- Modify: `lunar_lander_lab/utils/evaluation.py`

**Interfaces:**
- Consumes: `new_run_dir("benchmark")` from Task 3.
- Produces: `run_benchmark(controllers, env_name="LunarLander-v3", num_episodes=50, plot_path: Optional[str] = None) -> pd.DataFrame` (was `plot_path: str = "benchmark_results.png"`; default changed, override still works).

- [ ] **Step 1: Add paths import and change plot_path default**

Add to imports:
```python
from typing import Dict, Optional

from .paths import new_run_dir
```
(replaces the existing `from typing import Dict` line)

Change the function signature and its first lines:
```python
def run_benchmark(
    controllers: Dict[str, BaseController],
    env_name: str = "LunarLander-v3",
    num_episodes: int = 50,
    plot_path: Optional[str] = None,
) -> pd.DataFrame:
    """Evaluate each controller over the same set of episode seeds.

    Returns a DataFrame of per-controller summary metrics and writes a
    comparison bar chart to ``plot_path`` (default: a new
    runs/benchmark/<timestamp>/benchmark_results.png).
    """
    if plot_path is None:
        plot_path = str(new_run_dir("benchmark") / "benchmark_results.png")

    seeds = list(range(num_episodes))
```
The rest of the function (the loop building `rows`, `_plot_results` call) is unchanged.

- [ ] **Step 2: Verify**

```bash
python -m lunar_lander_lab.cli benchmark --episodes 5
```
Expected: prints the metrics table, then a new `runs/benchmark/<timestamp>/benchmark_results.png` exists.

- [ ] **Step 3: Commit**

```bash
git add lunar_lander_lab/utils/evaluation.py
git commit -m "Write benchmark charts to runs/benchmark/"
```

---

### Task 7: `pid_search.py` + `cli.py` — sweep output via `runs/pid_search/`

**Files:**
- Modify: `lunar_lander_lab/utils/pid_search.py`
- Modify: `lunar_lander_lab/cli.py`

**Interfaces:**
- Consumes: `new_run_dir("pid_search")` from Task 3.
- Produces: `run_monte_carlo(..., output_dir: Optional[str] = None, ...) -> pd.DataFrame` (was `output_dir: str = "pid_search_results"`).

- [ ] **Step 1: Add paths import and change output_dir default in pid_search.py**

Add to imports:
```python
from .paths import new_run_dir
```

Change the signature and body of `run_monte_carlo`:
```python
def run_monte_carlo(
    n_samples: int = 200,
    episodes_per_set: int = 30,
    seed: int = 0,
    env_name: str = "LunarLander-v3",
    param_space: Optional[Dict[str, Tuple[float, float]]] = None,
    output_dir: Optional[str] = None,
    n_jobs: Optional[int] = None,
) -> pd.DataFrame:
    """Latin-Hypercube-sample `param_space`, evaluate each set, save dataset + plots."""
    param_space = param_space or CORE_PARAM_SPACE
    gain_sets = sample_gain_sets(n_samples, param_space, seed=seed)
    work_items = [(gains, episodes_per_set, env_name) for gains in gain_sets]

    n_jobs = n_jobs or os.cpu_count() or 1
    results = []
    print(f"Evaluating {n_samples} gain sets x {episodes_per_set} episodes on {n_jobs} workers...")
    with multiprocessing.Pool(processes=n_jobs) as pool:
        for i, result in enumerate(pool.imap_unordered(_evaluate_gain_set, work_items), 1):
            results.append(result)
            if i % max(1, n_samples // 10) == 0 or i == n_samples:
                print(f"  [{i}/{n_samples}] evaluated")

    df = pd.DataFrame(results)
    out_dir = Path(output_dir) if output_dir else new_run_dir("pid_search")
    out_dir.mkdir(parents=True, exist_ok=True)
```
(The `out_dir.mkdir(...)` call is now redundant when `new_run_dir` already created the directory, but harmless and still required for the explicit-`output_dir` path — leave it as-is.) Everything below this point in the function (csv/json/plot writing) is unchanged.

- [ ] **Step 2: Change the CLI default for --output-dir**

In `lunar_lander_lab/cli.py`, the `pid-search` subparser:
```python
    pid_search_parser.add_argument("--output-dir", default="pid_search_results")
```
becomes:
```python
    pid_search_parser.add_argument("--output-dir", default=None)
```

- [ ] **Step 3: Verify**

```bash
python -m lunar_lander_lab.cli pid-search --samples 4 --episodes 2
```
Expected: prints progress, then `runs/pid_search/<timestamp>/` contains `pid_search_results.csv`, `sweep_manifest.json`, `best_gains.json`, `pid_search_scatter.png`, `pid_search_correlation.png`.

- [ ] **Step 4: Commit**

```bash
git add lunar_lander_lab/utils/pid_search.py lunar_lander_lab/cli.py
git commit -m "Write pid-search sweep output to runs/pid_search/"
```

---

### Task 8: Migrate existing local artifacts, untrack committed PNG

**Files:**
- Move: `lunar_lander_lab/models/ppo_lunar_lander.zip` → `runs/train/<timestamp>/ppo_lunar_lander.zip`
- Move: `lunar_lander_lab/pid_search_results_v2/*` → `runs/pid_search/v2/`
- Move: `lunar_lander_lab/pid_search_results_100826_0832/*` → `runs/pid_search/100826_0832/`
- Delete: `lunar_lander_lab/benchmark_results.png` (git-tracked)

None of these are git-tracked except the PNG, so the moves are plain filesystem operations (no `git mv`).

- [ ] **Step 1: Move the trained model into runs/train/**

```bash
mkdir -p runs/train/legacy_ppo_lunar_lander
mv lunar_lander_lab/models/ppo_lunar_lander.zip runs/train/legacy_ppo_lunar_lander/ppo_lunar_lander.zip
rmdir lunar_lander_lab/models
```

- [ ] **Step 2: Move the two pid-search result sets into runs/pid_search/**

```bash
mkdir -p runs/pid_search
mv lunar_lander_lab/pid_search_results_v2 runs/pid_search/v2
mv lunar_lander_lab/pid_search_results_100826_0832 runs/pid_search/100826_0832
```

- [ ] **Step 3: Untrack and delete the committed benchmark PNG**

```bash
git rm lunar_lander_lab/benchmark_results.png
```

- [ ] **Step 4: Verify**

```bash
ls runs/train runs/pid_search
git status --porcelain=v1 --ignored
```
Expected: `runs/` now holds the migrated files; `runs/` itself shows as ignored (`!!`); `lunar_lander_lab/benchmark_results.png` shows as staged for deletion (`D`).

- [ ] **Step 5: Commit**

```bash
git commit -m "Untrack committed benchmark_results.png (now generated under runs/)"
```

---

### Task 9: README rewrite

**Files:**
- Modify: `README.md` (repo root, moved here in Task 2)

- [ ] **Step 1: Rewrite README.md**

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "Rewrite README for new package layout and runs/ convention"
```

---

### Task 10: End-to-end smoke check

**Files:** none (verification only)

- [ ] **Step 1: Fresh install check**

```bash
pip install -e .
lunar-lander --help
```
Expected: help text, no errors.

- [ ] **Step 2: Run all four subcommands**

```bash
lunar-lander benchmark --episodes 3
lunar-lander pid-search --samples 4 --episodes 2
lunar-lander train --timesteps 2000
lunar-lander run --controller heuristic --seed 0
```
Expected: each completes without traceback; a heuristic episode renders and
prints a reward line matching pre-reorg behavior (same gains, same logic).

- [ ] **Step 3: Confirm runs/ is the only output location**

```bash
git status --porcelain=v1 --ignored
```
Expected: no new tracked-file changes from the commands above; `runs/`
listed as ignored; no stray files in `lunar_lander_lab/` or repo root.

- [ ] **Step 4: Confirm nothing references the old layout**

```bash
grep -rn "pid_search_results\"" lunar_lander_lab/ || true
grep -rn "models/" lunar_lander_lab/ || true
grep -rn "MODELS_DIR" lunar_lander_lab/ || true
```
Expected: no matches.

No commit — this task only verifies Tasks 1-9.
