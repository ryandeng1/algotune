from typing import Any, Dict
import numpy as np
import scipy.ndimage

class Solver:
    def __init__(self):
        self.mode = 'constant'
        self.order = 3
        self.reshape = False

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        image = problem["image"]
        angle = problem["angle"]

        # Handle multiples of 90° with fast NumPy routine
        if isinstance(angle, (int, float)) and angle % 90 == 0:
            k = int(round(angle / 90)) % 4
            rotated_image = np.rot90(image, k, axes=(0, 1))
        else:
            # Fallback to SciPy for arbitrary angles
            rotated_image = scipy.ndimage.rotate(
                image, angle, reshape=self.reshape, order=self.order, mode=self.mode
            )

        return {"rotated_image": rotated_image}