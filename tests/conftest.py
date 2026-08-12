"""Shared pytest setup.

Forces matplotlib onto a non-interactive backend before any test imports
pyplot. Several sweeps write plots as part of their normal run, and
matplotlib's default here is `tkagg` -- which needs a working Tk display and
fails intermittently under a full test run even though the same tests pass in
isolation. No test needs to draw on screen, so headless is both correct and
deterministic.
"""

import matplotlib

matplotlib.use("Agg")
