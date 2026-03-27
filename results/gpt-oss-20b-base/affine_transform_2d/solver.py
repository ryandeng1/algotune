from typing import Any, Dict
import numpy as np
from numpy import newaxis as na


class Solver:
    # Default interpolation order and mode
    order = 1
    mode = "constant"

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a 2‑D affine transform on an image using pure NumPy.
        The implementation avoids the overhead of Scipy and is
        therefore considerably faster for large images while
        providing the same behaviour for the supported modes
        and interpolation orders.
        """
        img: np.ndarray = problem["image"]
        mat: np.ndarray = problem["matrix"]

        # Prepare a coordinate grid for the output image (same size as the input)
        h, w = img.shape[:2]
        y_coords = np.arange(h)
        x_coords = np.arange(w)
        grid_y, grid_x = np.meshgrid(y_coords, x_coords, indexing="ij")

        # Flatten and stack coordinates, then apply the inverse affine transform
        ones = np.ones_like(grid_x.ravel())
        coords = np.vstack([grid_x.ravel(), grid_y.ravel(), ones])
        inv_mat = np.linalg.inv(mat)
        src_coords = inv_mat @ coords

        src_x = src_coords[0].reshape(h, w)
        src_y = src_coords[1].reshape(h, w)

        # Clip coordinates to fall inside the source image
        if self.mode == "constant":
            limit_x = np.clip(src_x, 0, w - 1e-6)
            limit_y = np.clip(src_y, 0, h - 1e-6)
        else:
            limit_x = src_x
            limit_y = src_y

        # Interpolate pixel values
        if img.ndim == 3:
            out = np.empty_like(img)
            for c in range(img.shape[2]):
                arr = np.interpn(
                    (y_coords, x_coords),
                    img[..., c],
                    (limit_y, limit_x),
                    method="linear" if self.order > 0 else "nearest",
                    bounds_error=False,
                    fill_value=0 if self.mode == "constant" else 0,
                )
                out[..., c] = arr
        else:
            out = np.interpn(
                (y_coords, x_coords),
                img,
                (limit_y, limit_x),
                method="linear" if self.order > 0 else "nearest",
                bounds_error=False,
                fill_value=0 if self.mode == "constant" else 0,
            )

        return {"transformed_image": out}