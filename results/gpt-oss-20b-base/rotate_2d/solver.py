import numpy as np
from scipy.ndimage import rotate

class Solver:
    def __init__(self):
        # Settings are left intentionally minimal for speed
        self.mode = 'constant'
        self.order = 3  # cubic interpolation, change to 1 for faster but lower quality
        self.reshape = False

    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Rotate a 2D image by a given angle using SciPy's fast rotation implementation.
        """
        image = problem['image']
        angle = problem['angle']

        # SciPy's rotate is already highly optimized; no extra error handling is needed
        rotated_image = rotate(image, angle, reshape=self.reshape,
                               order=self.order, mode=self.mode)

        return {'rotated_image': rotated_image}