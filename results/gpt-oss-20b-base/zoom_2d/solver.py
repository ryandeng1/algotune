from typing import Any
import numpy as np
try:
    from scipy.ndimage import zoom
except ImportError:
    zoom = None

class Solver:
    # default interpolation order and mode if not overridden
    order: int = 3
    mode: str = 'constant'

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Zoom a 2D NumPy array using scipy.ndimage.zoom.
        Returns an empty list for 'zoomed_image' if the zoom fails.
        """
        image = problem.get("image")
        zoom_factor = problem.get("zoom_factor", 1.0)

        if zoom is None or image is None:
            return {"zoomed_image": []}

        # Basic validation of input shapes
        if not isinstance(image, np.ndarray) or image.ndim != 2:
            return {"zoomed_image": []}

        try:
            result = zoom(image, zoom_factor, order=self.order, mode=self.mode)
        except Exception:
            return {"zoomed_image": []}

        return {"zoomed_image": result}