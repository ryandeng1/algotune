from typing import Any
import scipy.ndimage

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        image = problem["image"]
        matrix = problem["matrix"]
        
        order_val = self.order
        mode_val = self.mode
        
        try:
            transformed_image = scipy.ndimage.affine_transform(
                image, matrix, order=order_val, mode=mode_val
            )
        except Exception as e:
            return {"transformed_image": []}
        
        return {"transformed_image": transformed_image}