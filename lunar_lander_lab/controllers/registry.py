"""Controller registry.

Each controller module registers a name -> builder function, so `cli.py`
doesn't need a hardcoded if/elif chain or a controller-name list duplicated
across subparsers. Builders share one signature (`model_name`, `gains_path`)
even though most ignore most of it, because that's the actual union of
per-controller construction needs `cli.py` already had.

`mpc` and `lqr` are registered lazily: constructing either measures
constants off a live Gymnasium env (`MPCController`/`LQRController` both
default to `measure_planar_constants()`), and importing
`lunar_lander_lab.controllers` should stay free of that cost (see
`controllers/__init__.py`). Their builders only run once `.mpc`/`.lqr` is
imported, which `build_controller(...)` triggers on demand.
"""

import importlib
from typing import Callable, Dict

from .base import BaseController

_REGISTRY: Dict[str, Callable[..., BaseController]] = {}
_LAZY_MODULES: Dict[str, str] = {"mpc": ".mpc", "lqr": ".lqr"}


def register_controller(name: str):
    """Decorator: register a builder function under `name`."""

    def decorator(builder: Callable[..., BaseController]) -> Callable[..., BaseController]:
        _REGISTRY[name] = builder
        return builder

    return decorator


def controller_names() -> list:
    """All registered names, including lazily-loadable ones not yet imported."""
    return sorted(set(_REGISTRY) | set(_LAZY_MODULES))


def build_controller(name: str, **kwargs) -> BaseController:
    if name not in _REGISTRY and name in _LAZY_MODULES:
        importlib.import_module(_LAZY_MODULES[name], package=__package__)
    if name not in _REGISTRY:
        raise ValueError(f"Unknown controller: {name}")
    return _REGISTRY[name](**kwargs)
