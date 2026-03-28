import numpy as np
from scipy import ndimage

class Solver:
    def __init__(self):
        self.mode = "constant"
        self.order = 3

    def solve(self, problem: dict[str, any]) -> dict[str, any]:
        """
        Applies an affine transformation to a 2D image using scipy.ndimage.affine_transform.
        """
        image = np.asarray(problem["image"])
        matrix = np.asarray(problem["matrix"])
        transformed_image = ndimage.affine_transform(
            image, matrix, order=self.order, mode=self.mode
        )
        return {"transformed_image": transformed_image}