import numpy as np
import scipy.ndimage

class Solver:
    """
    Solver for 2D affine transform with cubic spline interpolation.

    The implementation uses scipy.ndimage.affine_transform to apply the given
    2x3 affine matrix to an N×N image.  The output has the same spatial
    dimensions as the input.  The transformation is performed with
    order=3 (cubic spline) and constant padding mode with value 0.
    """

    # Transformation parameters matching the reference
    order = 3
    mode = "constant"
    cval = 0.0

    def solve(self, problem: dict[str, any]) -> dict[str, any]:
        """
        Apply the affine transformation to the input image.

        Parameters
        ----------
        problem : dict
            Dictionary with keys:
                - "image": 2‑D array (list of lists or numpy array)
                - "matrix": 2×3 affine matrix (list of lists or numpy array)

        Returns
        -------
        dict
            Dictionary with key "transformed_image" containing the transformed
            image as a list of lists.  The shape of the output matches the
            input image shape.
        """

        # Convert inputs to numpy arrays
        image = np.asarray(problem["image"], dtype=float)
        matrix = np.asarray(problem["matrix"], dtype=float)

        # Affine transform: output shape is same as input
        transformed = scipy.ndimage.affine_transform(
            image,
            matrix,
            order=self.order,
            mode=self.mode,
            cval=self.cval,
            output_shape=image.shape,
            prefilter=True
        )

        # Convert result back to list of lists for consistency with reference
        return {"transformed_image": transformed.tolist()}
