import numpy as np
import scipy.ndimage

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        image = problem["image"]
        shift_vector = problem["shift"]
        shifted_image = scipy.ndimage.shift(
            image, shift_vector, order=3, mode='constant'
        )
        return {"shifted_image": shifted_image}
