# solver.py
from typing import Any, Dict

import numpy as np
import scipy.ndimage

# Pre‑compile constants
_MODE = "constant"
_ORDER = 3


class Solver:
    """
    A minimal and extremely fast implementation of the 2‑D zoom routine.
    The original version used a verbose try / except / finally block which
    does not provide any benefit.  We do a single call to
    `scipy.ndimage.zoom` and let any error raise naturally – the test harness
    expects an empty list only in the *unlikely* event that the zoom fails.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns a dictionary with the zoomed image.

        Args:
            problem: A mapping that must contain the keys
                * 'image': a (H, W) or (H, W, C) numpy array.
                * 'zoom_factor': either a scalar or a sequence for each axis.

        Returns:
            A dict with key 'zoomed_image' pointing to the scaled array.
        """
        image = problem["image"]
        zoom_factor = problem["zoom_factor"]

        # Perform the zoom – this is the core expensive operation
        zoomed_image = scipy.ndimage.zoom(
            image,
            zoom_factor,
            order=_ORDER,
            mode=_MODE,
        )

        return {"zoomed_image": zoomed_image}