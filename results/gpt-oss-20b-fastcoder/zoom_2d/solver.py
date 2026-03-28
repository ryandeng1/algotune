from typing import Any
import numpy as np
import scipy.ndimage

class Solver:
    """
    Solver for 2‑D image zoom using scipy.ndimage.zoom.
    """

    __slots__ = ("mode", "order")

    def __init__(self):
        # These parameters are fixed for all calls – no per‑call allocation.
        self.mode = "constant"
        self.order = 3

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Perform a zoom operation on a 2‑D numpy array.

        Parameters
        ----------
        problem : dict
            Must contain the keys:
            - "image": 2‑D numpy array or array‑like object.
            - "zoom_factor": float or sequence of 2 floats specifying the zoom for each axis.

        Returns
        -------
        dict
            Dictionary with a single key "zoomed_image" containing
            the resampled image.
        """
        # Extract data – rely on numpy coercion for speed.
        image = np.asarray(problem["image"])
        zoom_factor = np.asarray(problem["zoom_factor"], dtype=float)

        # Fast path – single scalar zoom on both axes.
        if zoom_factor.ndim == 0:
            zoom_factor = (zoom_factor, zoom_factor)

        # Perform the zoom.  This is the core of the solver,
        # and doing it directly avoids Python overhead.
        zoomed_image = scipy.ndimage.zoom(
            image,
            zoom_factor,
            order=self.order,
            mode=self.mode,
        )

        return {"zoomed_image": zoomed_image}