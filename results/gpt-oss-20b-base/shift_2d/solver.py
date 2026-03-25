import numpy as np
import scipy.ndimage

class Solver:
    order = 3
    mode = 'constant'

    def solve(self, problem, **kwargs):
        image = problem["image"]
        shift_vector = problem["shift"]

        # Use scipy's shift which is fast and matches the reference
        shifted_image = scipy.ndimage.shift(
            image,
            shift_vector,
            order=self.order,
            mode=self.mode,
        )
        return {"shifted_image": shifted_image}
