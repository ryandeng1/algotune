# solver.py

import numpy as np
import scipy.ndimage

class Solver:
    """
    Solver for the 2D image rotation task.
    Uses scipy.ndimage.rotate with the same parameters as the reference implementation
    to produce a cubic spline (order=3) rotated image with constant boundary padding.
    """

    # Parameters that match the reference implementation
    reshape = False
    order = 3
    mode = 'constant'
    cval = 0.0

    def solve(self, problem: dict) -> dict:
        """
        Rotate a square 2D image counter‑clockwise by a given angle.
        The output image has the same shape as the input.

        Parameters
        ----------
        problem : dict
            Must contain keys 'image' (n×n array-like of floats) and 'angle' (float).

        Returns
        -------
        dict
            Dictionary with key 'rotated_image' containing the
            rotated image as a NumPy array of shape (n, n).
        """
        # Extract input
        image = problem.get("image")
        angle = problem.get("angle")

        # Validate input types
        if image is None or angle is None:
            return {"rotated_image": []}

        # Convert to NumPy array of floats
        try:
            arr = np.asarray(image, dtype=np.float64)
        except Exception:
            return {"rotated_image": []}

        # Ensure square shape
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            return {"rotated_image": []}

        # Perform rotation
        try:
            rotated = scipy.ndimage.rotate(
                arr,
                angle,
                reshape=self.reshape,
                order=self.order,
                mode=self.mode,
                cval=self.cval
            )
        except Exception:
            return {"rotated_image": []}

        # Safety check: ensure no NaNs/Infs
        if not np.isfinite(rotated).all():
            # Fallback to zero array of correct shape
            rotated = np.zeros_like(arr)

        return {"rotated_image": rotated}
