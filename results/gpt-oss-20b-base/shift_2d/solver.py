from typing import Any, Dict
import numpy as np
import scipy.ndimage

class Solver:
    def __init__(self):
        self.mode = 'constant'
        self.order = 3

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Shift a 2‑D image by a (dy, dx) vector using cubic interpolation.
        """
        image = problem["image"]
        shift = problem["shift"]

        # SciPy's shift is fast and handles sub‑pixel offsets
        shifted = scipy.ndimage.shift(
            image,
            shift,
            order=self.order,
            mode=self.mode
        )

        return {"shifted_image": shifted}