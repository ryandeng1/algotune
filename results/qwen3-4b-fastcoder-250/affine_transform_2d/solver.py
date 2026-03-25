import numpy as np
import scipy.ndimage

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        image = problem["image"]
        matrix = problem["matrix"]
        transformed_image = scipy.ndimage.affine_transform(
            image, matrix, order=3, mode='constant', cval=0.0
        )
        return {"transformed_image": transformed_image}
