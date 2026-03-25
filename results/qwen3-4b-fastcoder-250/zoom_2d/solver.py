import numpy as np
import scipy.ndimage

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        image = problem["image"]
        zoom_factor = problem["zoom_factor"]
        zoomed_image = scipy.ndimage.zoom(image, zoom_factor, order=3, mode='constant')
        return {"zoomed_image": zoomed_image}
