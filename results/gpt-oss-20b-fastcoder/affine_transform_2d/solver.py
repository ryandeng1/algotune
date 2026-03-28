import numpy as np
import scipy.ndimage

class Solver:

    __slots__ = ("order", "mode")

    def __init__(self):
        # Use a fast interpolation order (linear)
        self.order = 1
        # Boundary handling: constant with zero padding is fast
        self.mode = "constant"

    def solve(self, problem: dict[str, any]) -> dict[str, any]:
        """
        Apply a 2‑D affine transformation using SciPy.
        """
        image = problem["image"]
        matrix = problem["matrix"]
        # Convert to numpy array if needed
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix)

        # Direct call – no try/except overhead
        transformed = scipy.ndimage.affine_transform(
            image,
            matrix,
            order=self.order,
            mode=self.mode,
        )
        return {"transformed_image": transformed}