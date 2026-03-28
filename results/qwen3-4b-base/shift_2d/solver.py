from typing import Any
import scipy.ndimage

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        image = problem["image"]
        shift_vector = problem["shift"]
        order = self.order
        mode = self.mode
        try:
            shifted_image = scipy.ndimage.shift(
                image, shift_vector, order=order, mode=mode
            )
        except Exception as e:
            return {"shifted_image": []}
        return {"shifted_image": shifted_image}