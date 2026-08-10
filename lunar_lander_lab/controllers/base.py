"""Abstract base class defining the controller interface."""

from abc import ABC, abstractmethod
from typing import Sequence


class BaseController(ABC):
    """Standard interface every controller (heuristic or learned) must implement."""

    @abstractmethod
    def get_action(self, observation: Sequence[float]) -> int:
        """Map an environment observation to a discrete action in [0, 1, 2, 3]."""
        raise NotImplementedError
