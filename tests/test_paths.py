"""Migrated from paths.py's __main__ self-check."""

import pytest

from lunar_lander_lab.utils.paths import latest_run_file, new_run_dir


def test_new_run_dir_creates_a_timestamped_directory(tmp_path):
    run_dir = new_run_dir("demo", base=tmp_path)
    assert run_dir.is_dir()
    assert run_dir.parent == tmp_path / "demo"


def test_latest_run_file_finds_the_file(tmp_path):
    run_dir = new_run_dir("demo", base=tmp_path)
    (run_dir / "result.txt").write_text("hello")

    found = latest_run_file("demo", "result.txt", base=tmp_path)
    assert found == run_dir / "result.txt"
    assert found.read_text() == "hello"


def test_latest_run_file_picks_the_newest_by_timestamp_name(tmp_path):
    """Run dirs sort by their timestamp names, so the newest must win."""
    for stamp in ("20260101_000000", "20260609_120000", "20260301_000000"):
        d = tmp_path / "demo" / stamp
        d.mkdir(parents=True)
        (d / "result.txt").write_text(stamp)

    assert latest_run_file("demo", "result.txt", base=tmp_path).read_text() == "20260609_120000"


def test_latest_run_file_raises_when_nothing_matches(tmp_path):
    with pytest.raises(FileNotFoundError):
        latest_run_file("missing", "x.txt", base=tmp_path)
