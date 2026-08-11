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
