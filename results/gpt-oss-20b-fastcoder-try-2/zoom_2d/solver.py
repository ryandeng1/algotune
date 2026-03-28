from typing import Any
import scipy.ndimage

class Solver:
    def __init__(self):
        self.mode = 'constant'
        self.order = 3

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        image = problem['image']          # 2‑D numpy array
        zoom_factor = problem['zoom_factor']
        try:
            zoomed = scipy.ndimage.zoom(
                image,
                zoom_factor,
                order=self.order,
                mode=self.mode
            )
        except Exception:
            return {'zoomed_image': []}
        return {'zoomed_image': zoomed}