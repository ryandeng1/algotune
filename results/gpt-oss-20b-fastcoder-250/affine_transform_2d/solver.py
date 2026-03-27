from typing import Any
import numpy as np
from scipy import ndimage

class Solver:
    # Default parameters for affine_transform
    order: int = 3   # cubic interpolation
    mode: str = "constant"  # out-of-bounds handling

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Performs a 2‑D affine transformation using scipy.ndimage.

        Parameters
        ----------
        problem : dict
            Must contain keys
            - "image" : 2‑D NumPy array.
            - "matrix" : 2×2 NumPy array or list of lists.

        Returns
        -------
        dict
            Dictionary with key "transformed_image" containing the
            rotated/translated image.
        """
        image: np.ndarray = problem["image"]
        matrix: np.ndarray = np.asarray(problem["matrix"], dtype=float)

        # Compute the inverse matrix once – ﻿affine_transform expects
        # the matrix that maps output coordinates to input coordinates.
        inv_matrix = np.linalg.inv(matrix)

        # Determine the output shape (same as input for simplicity)
        out_shape = image.shape

        # Use scipy's efficient C implementation
        transformed_image = ndimage.affine_transform(
            image,
            inv_matrix,
            output_shape=out_shape,
            order=self.order,
            mode=self.mode,
        )

        return {"transformed_image": transformed_image}