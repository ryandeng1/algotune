import numpy as np
import scipy.ndimage

class Solver:
    """Zoom a 2D image using cubic spline interpolation with zero padding."""

    # use the same parameters as the reference implementation
    order = 3
    mode = "constant"

    def solve(self, problem: dict) -> dict:
        """
        Parameters
        ----------
        problem : dict
            Must contain keys 'image' (2D list/array) and 'zoom_factor' (float).

        Returns
        -------
        dict
            Dictionary with a single key 'zoomed_image' containing the
            zoomed image as a 2D numpy array.
        """
        # Extract inputs
        image = problem.get("image")
        zoom_factor = problem.get("zoom_factor")

        # Convert to numpy array; rely on scipy to handle lists gracefully
        arr = np.asarray(image, dtype=float)

        # Perform the zoom operation
        zoomed = scipy.ndimage.zoom(
            arr, zoom_factor, order=self.order, mode=self.mode
        )

        # Return as list of lists to match expected JSON-like format
        return {"zoomed_image": zoomed.tolist()}
