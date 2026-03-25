import numpy as np
from numba import njit

@njit
def cubic_spline_2d(x, y, image):
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    x0 = np.clip(x0, 0, image.shape[0] - 1)
    y0 = np.clip(y0, 0, image.shape[1] - 1)
    x1 = np.clip(x1, 0, image.shape[0] - 1)
    y1 = np.clip(y1, 0, image.shape[1] - 1)
    v00 = image[y0, x0]
    v01 = image[y0, x1]
    v10 = image[y1, x0]
    v11 = image[y1, x1]
    x_val = x - x0
    y_val = y - y0
    t = x_val
    u = y_val
    v0 = v00 + (v01 - v00) * t + (v10 - 2*v00 + v01) * t**2 + (v11 - 3*v10 + 3*v00 - v01) * t**3
    v1 = v00 + (v10 - v00) * u + (v11 - 2*v10 + v00) * u**2 + (v11 - 3*v10 + 3*v00 - v01) * u**3
    return v0 + (v1 - v0) * u

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        image = problem["image"]
        matrix = problem["matrix"]
        n = image.shape[0]
        a, b, c = matrix[0]
        d, e, f = matrix[1]
        det = a * e - b * d
        transformed = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                x = (e * (i - c) - d * (j - f)) / det
                y = (a * (j - f) - d * (i - c)) / det
                x = np.clip(x, 0, n - 1)
                y = np.clip(y, 0, n - 1)
                transformed[i, j] = cubic_spline_2d(x, y, image)
        return {"transformed_image": transformed}
