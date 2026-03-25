import numpy as np
from scipy.ndimage import rotate

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        image = problem["image"]
        angle = problem["angle"]
        return {"rotated_image": rotate(image, angle, reshape=False, order=3, mode='constant')}
