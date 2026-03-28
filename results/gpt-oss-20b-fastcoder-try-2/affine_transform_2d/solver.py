import numpy as np
import scipy.ndimage

class Solver:
    def __init__(self):
        # pre‑set parameters for the affine transform
        self.order = 3
        self.mode = "constant"
        self.cval = 0.0  # default value for constant mode

    def solve(self, problem: dict) -> dict:
        """
        Apply a 2‑D affine transformation to the given image using
        SciPy's ndimage.affine_transform.

        Parameters
        ----------
        problem : dict
            Must contain:
                - 'image' : 2‑D NumPy array
                - 'matrix': 2x2 NumPy array or list representing the affine
                            matrix (row‑major order)

        Returns
        -------
        dict
            {'transformed_image': ndarray}
        """
        image = problem.get("image")
        matrix = problem.get("matrix")

        # Quick validation – if required data is missing, return empty array
        if image is None or matrix is None:
            return {"transformed_image": np.array([], dtype=np.uint8)}

        # Ensure inputs are NumPy arrays of the right dtype
        image = np.asarray(image)
        matrix = np.asarray(matrix, dtype=image.dtype)

        # Perform the transform
        transformed = scipy.ndimage.affine_transform(
            image,
            matrix,
            order=self.order,
            mode=self.mode,
            cval=self.cval,
        )

        return {"transformed_image": transformed}