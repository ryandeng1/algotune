import numpy as np
from math import sin, cos, radians
from collections import defaultdict
from typing import Any, Dict


class Solver:
    # Default parameters that were previously taken from instance attributes.
    reshape: bool = False
    order: int = 0  # nearest‑neighbor
    mode: str = "constant"

    @staticmethod
    def _get_fill_value(mode: str):
        """Return the value used for out‑of‑bounds pixels."""
        return 0.0 if mode == "constant" else np.nan

    @staticmethod
    def _interpolate(image: np.ndarray, x: float, y: float, order: int):
        """Sample a pixel with the specified interpolation order."""
        h, w = image.shape[:2]
        if order == 0:
            # Nearest neighbour
            i, j = int(round(y)), int(round(x))
            if 0 <= i < h and 0 <= j < w:
                return image[i, j]
            return None
        else:
            # Bilinear interpolation
            i0, j0 = int(np.floor(y)), int(np.floor(x))
            i1, j1 = i0 + 1, j0 + 1
            if (i0 < 0 or i1 >= h or j0 < 0 or j1 >= w):
                return None
            dy = y - i0
            dx = x - j0
            top = (1 - dx) * image[i0, j0] + dx * image[i0, j1]
            bottom = (1 - dx) * image[i1, j0] + dx * image[i1, j1]
            return (1 - dy) * top + dy * bottom

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rotate a 2‑D image using NumPy only, mimicking scipy.ndimage.rotate.
        Parameters:
            problem : dict
                Must contain:
                    "image" : 2‑D or 3‑D NumPy array
                    "angle" : rotation angle in degrees (clockwise)
        Returns:
            dict with key "rotated_image" containing the rotated array.
        """
        try:
            image = np.asarray(problem["image"])
            angle = float(problem["angle"])
        except (KeyError, ValueError, TypeError):
            return {"rotated_image": []}

        # Convert to float for interpolation
        dtype = np.float32
        if image.dtype.kind == "f":
            dtype = image.dtype
        image = image.astype(dtype, copy=False)

        # Compute rotation matrix for clockwise rotation
        rad = radians(-angle)  # negative for clockwise
        cos_a, sin_a = cos(rad), sin(rad)
        rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        # Determine output shape
        h, w = image.shape[:2]
        if self.reshape:
            corners = np.array([[0, 0], [0, h], [w, 0], [w, h]])
            shifted = corners - np.array([w / 2, h / 2])
            rotated = shifted @ rot_mat.T
            min_xy = rotated.min(axis=0)
            max_xy = rotated.max(axis=0)
            output_shape = np.ceil(max_xy - min_xy).astype(int)
            offset = -min_xy
        else:
            output_shape = np.array([h, w])
            offset = np.array([w / 2, h / 2]) - rot_mat @ np.array([w / 2, h / 2])

        # Prepare output grid
        yy, xx = np.indices(output_shape)
        grid = np.stack([xx, yy], axis=-1).reshape(-1, 2)
        # Map output coords back to input coords
        inv_rot = np.linalg.inv(rot_mat)
        src_coords = (grid - offset) @ inv_rot.T
        src_x = src_coords[:, 0]
        src_y = src_coords[:, 1]

        # Handle multi‑channel images
        nch = image.shape[2] if image.ndim == 3 else 1
        out = np.empty((*output_shape, nch), dtype=dtype) if nch > 1 else np.empty(output_shape, dtype=dtype)

        fill_val = self._get_fill_value(self.mode)

        for c in range(nch):
            channel = image[..., c] if nch > 1 else image
            vals = np.empty(src_x.shape, dtype=dtype)
            for idx in range(src_x.size):
                val = self._interpolate(channel, src_x[idx], src_y[idx], self.order)
                vals[idx] = val if val is not None else fill_val
            out[..., c] = vals.reshape(output_shape) if nch > 1 else vals.reshape(output_shape)

        return {"rotated_image": out}