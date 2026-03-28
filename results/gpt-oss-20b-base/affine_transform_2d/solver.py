import numpy as np
from typing import Any

class Solver:
    """
    Simple, fast 2‑D affine transformer using only NumPy.
    The implementation supports only affine matrices of shape (2, 3) and
    produces nearest‑neighbour results.
    """

    def __init__(self, order: int = 0, mode: str = 'constant', cval: float = 0.0) -> None:
        self.order = order               # nearest neighbour only
        self.mode = mode
        self.cval = cval

    def _map_grid(self, in_shape: tuple[int, int], matrix: np.ndarray) -> np.ndarray:
        """
        Compute the inverse mapping from output pixels to input coordinates.
        """
        h, w = in_shape
        # Create output grid
        y, x = np.indices((h, w))
        # Homogeneous coordinates
        ox, oy = x.ravel(), y.ravel()
        ones = np.ones_like(ox)
        out_coords = np.vstack((ox, oy, ones))

        inv_mat = np.linalg.pinv(matrix)  # (3,2) returning inverse
        in_coords = inv_mat @ out_coords  # (3, N)
        return in_coords[:2].reshape((2, h, w))

    def _sample(self, image: np.ndarray, in_x: np.ndarray, in_y: np.ndarray) -> np.ndarray:
        """
        Nearest‑neighbour sampling with constant boundary handling.
        """
        h, w = image.shape[:2]
        if image.ndim == 3:
            channels = image.shape[2]
        else:
            channels = 1
            image = image[:, :, np.newaxis]

        x = np.round(in_x).astype(int)
        y = np.round(in_y).astype(int)

        # Outside indices
        mask = ((x < 0) | (x >= w) | (y < 0) | (y >= h))

        # Clip indices for valid access
        x_clipped = np.clip(x, 0, w - 1)
        y_clipped = np.clip(y, 0, h - 1)

        out = image[y_clipped, x_clipped]
        if self.mode == 'constant':
            out[mask] = self.cval
        elif self.mode == 'nearest':
            out[mask] = image[y_clipped[mask], x_clipped[mask]]  # already nearest

        if channels == 1:
            return out.squeeze()
        return out

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        image = problem['image']
        matrix = np.asarray(problem['matrix'], dtype=float)

        if matrix.shape != (2, 3):
            raise ValueError("Matrix must be 2x3 for 2D affine transform")

        # Compute mapping for the whole output grid
        in_x, in_y = self._map_grid(image.shape, np.vstack((matrix, [0, 0, 1])))
        transformed = self._sample(image, in_x, in_y)
        return {'transformed_image': transformed.tolist()}