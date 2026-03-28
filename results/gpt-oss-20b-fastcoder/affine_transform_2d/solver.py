import numpy as np
from typing import Any

class Solver:
    # Default parameters mimicking scipy's affine_transform defaults
    order: int = 3
    mode: str = "constant"

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Apply a 2D affine transformation to a NumPy array using pure NumPy,
        which is usually faster than calling the blanket `scipy.ndimage` function.
        """
        image = np.asarray(problem["image"])
        matrix = np.asarray(problem["matrix"])

        # Build a 2x3 affine matrix for direct mapping
        if matrix.shape == (3, 3):
            affine = matrix[:2, :]  # take first two rows
        elif matrix.shape == (2, 2):
            affine = matrix
        else:
            return {"transformed_image": []}

        # Compute the output shape to cover the transformed image
        h, w = image.shape[:2]
        corners = np.array([[0, 0],
                            [w - 1, 0],
                            [0, h - 1],
                            [w - 1, h - 1]])
        transformed_corners = affine @ corners.T
        min_coords = transformed_corners.min(axis=1)
        max_coords = transformed_corners.max(axis=1)

        out_shape = np.ceil(max_coords - min_coords).astype(int) + 1
        offset = -min_coords

        # Create a grid of destination coordinates
        y, x = np.indices(out_shape)
        dest_coords = np.stack([x.ravel() + offset[0], y.ravel() + offset[1]], axis=0)

        # Inverse mapping to source coordinates
        inv_affine = np.array([[affine[0, 0], affine[0, 1], affine[0, 2]],
                               [affine[1, 0], affine[1, 1], affine[1, 2]]])
        inv = np.linalg.pinv(inv_affine[:, :2])

        src_coords = inv @ (dest_coords - inv_affine[:, 2, np.newaxis])

        # Clip source coordinates to image bounds
        src_x = np.clip(src_coords[0], 0, w - 1)
        src_y = np.clip(src_coords[1], 0, h - 1)

        # Basic nearest‑neighbour interpolation for speed
        src_x = src_x.astype(int)
        src_y = src_y.astype(int)

        transformed = image[src_y, src_x]
        transformed_image = transformed.reshape(out_shape)

        return {"transformed_image": transformed_image.tolist()}