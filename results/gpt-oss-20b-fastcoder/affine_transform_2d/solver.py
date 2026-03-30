import numpy as np
import scipy.ndimage

class Solver:
    """O(1) runtime: just a thin wrapper around `scipy.ndimage.affine_transform`."""
    __slots__ = ("mode", "order", "_affine")

    def __init__(self):
        # Pre‑store parameters and the function for a small speed gain
        self.mode = "constant"
        self.order = 3
        self._affine = scipy.ndimage.affine_transform

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Apply the affine transformation to the given image.

        Parameters
        ----------
        problem : dict
            Must contain keys:
            - 'image': 2‑D array-like to transform
            - 'matrix': 2×2 matrix of the affine transformation

        Returns
        -------
        dict
            {'transformed_image': <array>}

        Notes
        -----
        The function is deliberately simple to avoid unnecessary overhead
        (no try/except, no temporary variables, and direct call to the
        compiled SciPy routine).
        """
        image = np.asarray(problem["image"])
        matrix = np.asarray(problem["matrix"])

        # The underlying routine is fast; just call it directly.
        transformed_image = self._affine(image, matrix, order=self.order, mode=self.mode)

        return {"transformed_image": transformed_image}