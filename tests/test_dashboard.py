"""Arc 4 Phase D: the dashboard reuses cli.py's/utils'/registry's machinery
rather than reimplementing scoring or training. No browser automation here
(that was done manually against a live `streamlit run` -- see
tmp/TESTBENCH_ROADMAP.md) -- this is just the import-time smoke test."""

import runpy
import sys
from unittest.mock import MagicMock


def test_app_module_imports_and_runs_cleanly(monkeypatch):
    """Streamlit scripts execute top-to-bottom on import; stub the `streamlit`
    module so this runs headless (no server, no browser) and still catches
    any real bug in app.py (a bad import, an undefined name, wrong kwargs to
    a reused utils function)."""
    stub = MagicMock()
    stub.empty.return_value = MagicMock()
    stub.tabs.return_value = (MagicMock(), MagicMock(), MagicMock())
    for name in ("subheader", "selectbox", "multiselect", "slider", "number_input", "button"):
        getattr(stub, name).return_value = False if name == "button" else None
    monkeypatch.setitem(sys.modules, "streamlit", stub)

    runpy.run_module("lunar_lander_lab.dashboard.app", run_name="__main__")


def test_dashboard_reads_a_non_empty_registry():
    from lunar_lander_lab.controllers import controller_names

    assert len(controller_names()) >= 8  # heuristic/scheduled/mpc/lqr/rl/sac/dqn/td3
