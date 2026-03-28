from typing import Any
import numpy as np
from scipy.ndimage import zoom

class Solver:
    __slots__ = ("order", "mode")

    def __init__(self):
        self.order = 3
        self.mode = "constant"

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Extract inputs
        image = problem["image"]
        zoom_factor = problem["zoom_factor"]

        # Compute zoomed image
        # scipy.ndimage.zoom handles float and int zoom factors.
        zoomed_image = zoom(image, zoom_factor, order=self.order, mode=self.mode)

        return {"zoomed_image": zoomed_image}