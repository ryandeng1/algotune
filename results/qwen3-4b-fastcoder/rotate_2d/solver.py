from typing import Any
import scipy.ndimage

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        image = problem["image"]
        angle = problem["angle"]
        reshape = self.reshape
        order = self.order
        mode = self.mode
        try:
            rotated_image = scipy.ndimage.rotate(image, angle, reshape=reshape, order=order, mode=mode)
        except Exception as e:
            return {"rotated_image": []}
        return {"rotated_image": rotated_image}