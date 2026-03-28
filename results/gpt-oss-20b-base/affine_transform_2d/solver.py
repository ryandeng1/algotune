import numpy as np
import scipy.ndimage

class Solver:
    def __init__(self):
        # Pre‑define the common parameters for the transform
        self.mode = 'constant'
        self.order = 3

    def solve(self, problem: dict[str, any]) -> dict[str, any]:
        """
        Apply a 2‑D affine transformation to the input image using
        scipy.ndimage.affine_transform. The result is returned as a
        dictionary containing the transformed image array.
        """
        image = problem["image"]
        matrix = problem["matrix"]

        # `affine_transform` expects the transformation matrix that maps
        # output coordinates to input coordinates.  We assume the provided
        # matrix already satisfies this specification.
        transformed_image = scipy.ndimage.affine_transform(
            image,
            matrix,
            order=self.order,
            mode=self.mode,
        )

        return {"transformed_image": transformed_image}