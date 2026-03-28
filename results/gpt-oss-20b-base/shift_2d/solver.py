from typing import Any
import numpy as np
from scipy.ndimage import shift


class Solver:
    """
    Optimized solver for shifting 2D images using scipy.ndimage.shift.
    """

    __slots__ = ("mode", "order")

    def __init__(self):
        # Keep the mode/order as attributes to avoid repeated allocation
        self.mode = "constant"
        self.order = 3

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solves a 2‑D image shift problem.

        Parameters
        ----------
        problem : dict
            Must contain:
                - 'image'   : 2‑D NumPy array to shift
                - 'shift'   : 2‑tuple (shift_y, shift_x) or list of two floats

        Returns
        -------
        dict
            {'shifted_image': ndarray}
        """
        image = problem["image"]
        # Ensure the shift vector is a tuple of floats
        shift_vector = tuple(map(float, problem["shift"]))
        # The core of the algorithm – a single C‑level call
        shifted = shift(image, shift_vector, order=self.order, mode=self.mode)
        return {"shifted_image": shifted}