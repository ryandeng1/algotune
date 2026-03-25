# solver.py
import numpy as np
from typing import Any, Dict
import scipy.ndimage

class Solver:
    """
    Zoom a 2D image by a given factor using cubic spline interpolation
    (order=3) and zero padding outside the image boundaries.
    """

    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Parameters
        ----------
        problem : dict
            Must contain:
                "image"      : 2D sequence (list of lists or numpy array)
                "zoom_factor": float, zoom multiplier (same for both axes)

        Returns
        -------
        dict
            Key "zoomed_image" with the zoomed image as a numpy array.
            Shape matches scipy.ndimage.zoom with the same parameters.
        """
        # Extract values
        image = np.asarray(problem["image"], dtype=float)
        zoom_factor = float(problem["zoom_factor"])

        # Perform the zoom with cubic spline and constant mode
        zoomed_image = scipy.ndimage.zoom(
            image,
            zoom=zoom_factor,
            order=3,
            mode="constant",
            cval=0.0,
        )

        return {"zoomed_image": zoomed_image}
