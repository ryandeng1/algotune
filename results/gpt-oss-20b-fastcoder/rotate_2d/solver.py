import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Rotates a 2D image by the given angle using NumPy-only routines.
        Works best for multiples of 90°, otherwise uses a simple bilinear interpolation.

        :param problem: A dictionary with keys:
                        - 'image': 2D NumPy array
                        - 'angle': rotation angle in degrees
        :return: A dictionary with key "rotated_image" containing the rotated array.
        """
        image = problem["image"]
        angle = problem["angle"] % 360  # normalize

        # 1. Exact 90° multiples: use rot90 which is ultra‑fast
        if angle % 90 == 0:
            k = int(angle // 90) % 4
            rotated = np.rot90(image, k=k)
            return {"rotated_image": rotated}

        # 2. General case: affine transform with bilinear interpolation
        # compute rotation matrix
        theta = np.deg2rad(angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

        # image centre
        h, w = image.shape[:2]
        cx, cy = w / 2, h / 2

        # prepare output canvas size (same as input to avoid reshape)
        out = np.zeros_like(image)

        # iterate over output pixel coordinates and map back
        for y in range(h):
            for x in range(w):
                # relative coordinate
                xr, yr = x - cx, y - cy
                # inverse map
                xi, yi = mat.T @ np.array([xr, yr])
                xi, yi = xi + cx, yi + cy
                # bilinear interpolation
                if 0 <= xi < w - 1 and 0 <= yi < h - 1:
                    ix, iy = int(np.floor(xi)), int(np.floor(yi))
                    dx, dy = xi - ix, yi - iy
                    # fetch 4 neighbours
                    a = image[iy, ix]
                    b = image[iy, ix + 1]
                    c = image[iy + 1, ix]
                    d = image[iy + 1, ix + 1]
                    out[y, x] = (
                        a * (1 - dx) * (1 - dy) +
                        b * dx * (1 - dy) +
                        c * (1 - dx) * dy +
                        d * dx * dy
                    )
        return {"rotated_image": out}