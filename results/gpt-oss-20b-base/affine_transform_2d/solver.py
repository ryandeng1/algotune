import numpy as np
import scipy.ndimage

class Solver:
    """
    Solver for 2D affine transform using SciPy's ndimage.affine_transform.
    Applies cubic spline interpolation (order=3) with constant boundary padding (0).
    """
    def __init__(self):
        # Constants for the affine transformation
        self.order = 3
        self.mode = 'constant'
        self.cval = 0.0

    def solve(self, problem, **kwargs) -> dict:
        """
        Apply 2D affine transformation to the input image.

        Parameters
        ----------
        problem : dict
            Must contain:
                - "image": (n, n) array-like of floats.
                - "matrix": (2, 3) array-like transformation matrix.

        Returns
        -------
        dict
            {
                "transformed_image": (n, n) numpy array of floats
            }
        """
        # Extract data
        image = np.asarray(problem["image"], dtype=float)
        matrix = np.asarray(problem["matrix"], dtype=float)

        # Perform affine transformation.
        # scipy.ndimage.affine_transform expects the inverse matrix, so we
        # supply the same matrix because the reference uses it directly.
        transformed = scipy.ndimage.affine_transform(
            image,
            matrix,
            order=self.order,
            mode=self.mode,
            cval=self.cval,
            prefilter=False  # faster for cubic (order=3) and small images
        )

        return {"transformed_image": transformed}
