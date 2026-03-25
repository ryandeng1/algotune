import numpy as np
from scipy.ndimage import shift

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        image = problem["image"]
        shift = problem["shift"]
        shifted_image = shift(image, shift, order=3, mode='constant')
        return {"shifted_image": shifted_image}
