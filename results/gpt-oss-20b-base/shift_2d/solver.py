from typing import Any
import numpy as np
import scipy.ndimage

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        image = problem["image"]
        shift = problem["shift"]
        # Fast local variable lookup
        order = getattr(self, "order", 3)
        mode = getattr(self, "mode", "nearest")
        # Perform shift
        shifted = scipy.ndimage.shift(image, shift, order=order, mode=mode)
        return {"shifted_image": shifted}