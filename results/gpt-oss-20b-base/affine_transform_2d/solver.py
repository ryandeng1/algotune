import numpy as np
from scipy.ndimage import affine_transform

class Solver:
    """
    Fast affine image transformation solver.
    """

    __slots__ = ("order", "mode")

    def __init__(self):
        # default parameters matching the original implementation
        self.order: int = 3
        self.mode: str = "constant"

    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Apply a 2‑D affine transformation to an image.

        Parameters
        ----------
        problem
            Dictionary with keys:
                * 'image'   : 2-D `ndarray` (image data).
                * 'matrix'  : 2‑D `ndarray` (2×2 transformation matrix).

        Returns
        -------
        dict
            ``{'transformed_image': <ndarray>}`` containing the transformed image.
        """
        image = problem["image"]
        matrix = problem["matrix"]

        # Ensure the image is C‑contiguous and in a suitable dtype.
        # Using float32 gives the best trade‑off between speed and precision.
        img = np.ascontiguousarray(image, dtype=np.float32)

        # Perform the affine transform.
        # We compute the inverse matrix directly for speed.
        transform = affine_transform(
            img,
            matrix,
            order=self.order,
            mode=self.mode,
            cval=0,
        )

        # Return the result with the original dtype.
        return {"transformed_image": transform.astype(image.dtype)}