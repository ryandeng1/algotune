from typing import Any
import scipy.ndimage

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        image = problem["image"]
        shift_vector = problem["shift"]
        order_val = self.order
        mode_val = self.mode

        try:
            shifted_image = scipy.ndimage.shift(image, shift_vector, order=order_val, mode=mode_val)
        except Exception:
            return {"shifted_image": []}
        
        return {"shifted_image": shifted_image}