"""Abstract base class defining the controller interface."""

from abc import ABC, abstractmethod
from typing import Sequence, Union

import numpy as np


class BaseController(ABC):
    """Standard interface every controller (heuristic or learned) must implement."""

    @abstractmethod
    def get_action(self, observation: Sequence[float]) -> Union[int, np.ndarray]:
        """Map an observation to an action.

        An int in [0, 1, 2, 3] for the default discrete `LunarLander-v3`, or a
        2-vector for the continuous variant (see tmp/SPEED_ROADMAP.md Phase
        8b). Which one a controller returns must match the action space of the
        env it is evaluated against.
        """
        raise NotImplementedError
