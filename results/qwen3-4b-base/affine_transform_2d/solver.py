from typing import Any
import numpy as np
import scipy.ndimage

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        image = problem["image"]
        matrix = problem["matrix"]
        
        try:
            # Calculate output shape using affine transformation properties
            h_out = int(np.ceil(image.shape[0] * matrix[0, 0] + image.shape[1] * matrix[0, 1]))
            w_out = int(np.ceil(image.shape[0] * matrix[1, 0] + image.shape[1] * matrix[1, 1]))
            output_shape = (h_out, w_out)
            
            # Pre-allocate output array to avoid memory allocation overhead
            output_array = np.zeros(output_shape, dtype=image.dtype)
            transformed_image = scipy.ndimage.affine_transform(
                image, matrix, order=self.order, mode=self.mode, output=output_array
            )
        except Exception as e:
            return {"transformed_image": []}
        
        return {"transformed_image": transformed_image}