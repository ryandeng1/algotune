import numpy as np
import scipy.ndimage


class Solver:
    """
    Fast implementation of 2D image rotation using scipy.ndimage.rotate.
    """

    def __init__(self):
        # Parameters for rotation
        self.reshape = False          # keep output same shape as input
        self.order = 3                # cubic spline interpolation
        self.mode = 'constant'        # pad with zeros
        self.cval = 0.0

    def solve(self, problem: dict) -> dict:
        """
        Rotate the image counter‑clockwise by the specified angle.

        Parameters
        ----------
        problem : dict
            Must contain keys:
                - "image": 2‑D array (list or np.ndarray) of floats.
                - "angle": float, rotation angle in degrees.

        Returns
        -------
        dict
            A dictionary with key "rotated_image" containing the rotated image
            as a list of lists (for consistency with the reference solution).
        """
        try:
            image = np.asarray(problem["image"], dtype=float)
            angle = float(problem["angle"])
        except Exception as e:
            # In case of malformed input, return empty result
            return {"rotated_image": []}

        try:
            rotated = scipy.ndimage.rotate(
                image,
                angle,
                reshape=self.reshape,
                order=self.order,
                mode=self.mode,
                cval=self.cval,
            )
        except Exception:
            return {"rotated_image": []}

        # Convert to nested lists to match expected output format
        return {"rotated_image": rotated.tolist()}
