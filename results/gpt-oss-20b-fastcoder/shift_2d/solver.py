import numpy as np
from typing import Any
import scipy.ndimage

class Solver:
    # Default parameters for the shift (can be overridden if needed)
    order: int = 1
    mode: str = 'reflect'

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Shift 2‑D array using SciPy with minimal overhead.
        """
        image = problem["image"]
        shift_vector = problem["shift"]
        # Perform shift directly – SciPy validates inputs
        shifted_image = scipy.ndimage.shift(
            image, shift_vector, order=self.order, mode=self.mode
        )
        return {"shifted_image": shifted_image}