from typing import Any
import scipy.ndimage

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        image = problem["image"]
        zoom_factor = problem["zoom_factor"]

        try:
            zoomed_image = scipy.ndimage.zoom(image, zoom_factor, order=self.order, mode=self.mode)
        except Exception:
            return {"zoomed_image": []}

        return {"zoomed_image": zoomed_image}