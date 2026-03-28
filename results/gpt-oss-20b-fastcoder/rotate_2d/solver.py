import numpy as np
import scipy.ndimage

class Solver:
    def __init__(self):
        self.mode = 'constant'
        self.order = 3
        self.reshape = False

    def _rot90_fast(self, im, k):
        """rotate by k*90 degrees using np.rot90"""
        return np.rot90(im, k=k)

    def solve(self, problem: dict) -> dict:
        image = problem['image']
        angle = problem['angle'] % 360  # normalize

        # handle common multiples of 90° directly with np.rot90
        if angle % 90 == 0:
            k = int(round(angle / 90))
            rotated_image = self._rot90_fast(image, k)
        else:
            # fallback to scipy for arbitrary angles
            rotated_image = scipy.ndimage.rotate(
                image,
                angle,
                reshape=self.reshape,
                order=self.order,
                mode=self.mode
            )

        return {'rotated_image': rotated_image}