# Lunar Lander Lab — repo reorg design

Date: 2026-08-10

## Purpose

Repo stays a single project (`lunar_lander_lab`). Currently run outputs
(`models/`, `pid_search_results_v2/`, `pid_search_results_100826_0832/`,
`benchmark_results.png`) accumulate loose in the project dir with
inconsistent naming, gains are hand-copied from sweep results into source
code, and the package can only run via `cd lunar_lander_lab && python
main.py`. This reorg fixes all three without changing any runtime behavior
(controllers, CLI commands, algorithms are untouched).

## Decisions (confirmed with user)

- Repo scope: stays single-project. No multi-project scaffolding.
- Run artifacts: one `runs/` directory, timestamped subfolders per run,
  kept forever, gitignored wholesale.
- Packaging: add `pyproject.toml` + console entry point (`lunar-lander`),
  installable with `pip install -e .`, runnable from anywhere.
- Heuristic gains: loaded from a config file (`configs/heuristic_gains.json`)
  instead of hardcoded class attributes with a manually-updated comment.
- `lunar_lander_lab/benchmark_results.png` (currently tracked in git):
  untrack it — future benchmark charts only live under gitignored
  `runs/benchmark/`.

## Target layout

```
Lunar/
├── .gitignore
├── pyproject.toml                    # deps + `lunar-lander` console script
├── README.md                         # moved up from lunar_lander_lab/
├── runs/                             # gitignored, all run output, kept forever
│   ├── train/<timestamp>/ppo_lunar_lander.zip
│   ├── pid_search/<timestamp>/{pid_search_results.csv, sweep_manifest.json,
│   │                            best_gains.json, *.png}
│   └── benchmark/<timestamp>/benchmark_results.png
└── lunar_lander_lab/
    ├── __init__.py                   # new — makes it a real package
    ├── cli.py                        # renamed from main.py, relative imports
    ├── configs/
    │   └── heuristic_gains.json      # new — current v2 tuned gains
    ├── controllers/
    │   ├── __init__.py
    │   ├── base.py
    │   ├── heuristic.py              # reads configs/heuristic_gains.json
    │   └── rl_agent.py               # saves/loads via runs/train/
    └── utils/
        ├── __init__.py
        ├── paths.py                  # new — RUNS_DIR, new_run_dir(), latest_run_file()
        ├── evaluation.py             # saves to runs/benchmark/
        └── pid_search.py             # saves to runs/pid_search/
```

## Components

**`utils/paths.py`** (new) — shared by train/pid-search/benchmark so all
three use one naming convention instead of three ad-hoc ones.

- `PACKAGE_ROOT = Path(__file__).resolve().parent.parent`
- `REPO_ROOT = PACKAGE_ROOT.parent`
- `RUNS_DIR = REPO_ROOT / "runs"`
- `new_run_dir(kind: str) -> Path` — creates and returns
  `RUNS_DIR / kind / <YYYYMMDD_HHMMSS>/`.
- `latest_run_file(kind: str, filename: str) -> Path` — globs
  `RUNS_DIR/kind/*/filename`, sorted (timestamp dirs sort lexically),
  returns the last match; raises `FileNotFoundError` if none exist.

Non-trivial logic (glob + sort + not-found branch) gets one `__main__`
assert-based self-check in the same file — no test framework added.

**`controllers/rl_agent.py`** — `MODELS_DIR` constant removed.

- `train()`: saves into `new_run_dir("train") / f"{save_path}.zip"` instead
  of a fixed `models/` dir.
- `load(model_name)`: signature unchanged. If `model_name` resolves to an
  existing path, load it directly (unchanged escape hatch for explicit
  `--model` paths). Otherwise resolve via
  `latest_run_file("train", f"{model_name}.zip")` — always picks the most
  recently trained checkpoint automatically.

**`controllers/heuristic.py`** — gains for the 5 swept parameters
(`ANGLE_GAIN_VEL`, `DESCENT_GAIN`, `TARGET_DESCENT_SPEED`,
`ANGLE_THRESHOLD`, `HOVER_THRESHOLD`) load from
`configs/heuristic_gains.json` at import time via `json.loads`. The 3
never-swept gains (`ANGLE_GAIN_POS`, `ANGLE_ERROR_GAIN`, `ANGULAR_VEL_GAIN`)
stay as hardcoded class attributes — they're not part of the tuning
workflow. No fallback-if-missing branch: the config file is committed to
the repo, so it always exists.

Promotion workflow after a sweep is manual, not automated: inspect
`runs/pid_search/<ts>/best_gains.json`, and if it's actually better,
hand-copy its `best_gains` values into `configs/heuristic_gains.json`. A
deliberate choice — auto-writing the config after every sweep could
silently regress the default if a run is smaller/noisier than the current
best.

**`utils/evaluation.py`** (`run_benchmark`) — `plot_path` default changes
from the fixed string `"benchmark_results.png"` to `None`; when `None`,
resolves to `new_run_dir("benchmark") / "benchmark_results.png"`. The
parameter stays overridable.

**`utils/pid_search.py`** (`run_monte_carlo`) — `output_dir` default
changes from `"pid_search_results"` to `None`; when `None`, resolves to
`new_run_dir("pid_search")`.

**`cli.py`** (renamed from `main.py`) — same commands/flags. Imports become
package-relative (`from .controllers import ...`, `from .utils.evaluation
import ...`). `pid-search` subcommand's `--output-dir` argparse default
changes from `"pid_search_results"` to `None` to match the new default
resolution.

**`pyproject.toml`** (new) — replaces `requirements.txt`.

```toml
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

**`.gitignore`** — remove the now-obsolete
`lunar_lander_lab/models` / `lunar_lander_lab/pid_search_results_*` lines;
add `runs/`, `*.egg-info/`, `build/`, `dist/`.

**`README.md`** — moved to repo root. Update setup instructions
(`pip install -e .` instead of `pip install -r requirements.txt`), project
structure diagram, CLI invocation (works from anywhere via `lunar-lander
...`, or still `python -m lunar_lander_lab.cli ...` from repo root), and a
short note on the gains-promotion workflow.

## Migration of existing local files (untracked, not in git)

- `lunar_lander_lab/models/ppo_lunar_lander.zip` → `runs/train/<ts>/`
- `lunar_lander_lab/pid_search_results_v2/` → `runs/pid_search/v2/`
  (kept as-is, not renamed to the timestamp convention — legacy dirs are
  history, only new runs follow the new naming)
- `lunar_lander_lab/pid_search_results_100826_0832/` →
  `runs/pid_search/100826_0832/` (same reasoning)

## Git-tracked change

`git rm --cached lunar_lander_lab/benchmark_results.png` — file stays on
disk locally (now gitignored under the new pattern set, or just deleted
since it's regenerable) but is no longer committed.

## Out of scope

- No tests added beyond the one `paths.py` self-check (not requested).
- No auto-promotion of gains from sweep to config (deliberately manual).
- No multi-project / `src/` layout (repo confirmed single-project).
- Controller logic, CLI command behavior, and algorithms are unchanged.
