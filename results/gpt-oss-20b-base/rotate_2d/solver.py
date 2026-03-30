from typing import Any
import numpy as np
import scipy.ndimage

class Solver:
    """
    Optimised solver for rotating 2D images using SciPy's ndimage.rotate.
    The implementation avoids unnecessary exception handling and attribute lookups
    inside the hot path. All configuration parameters are stored as attributes
    set once in __init__.
    """

    def __init__(self):
        # Rotation configuration – these are constant for all calls
        self.mode = "constant"
        self.order = 3
        self.reshape = False

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Rotates the input image by the specified angle.
        The function assumes that the 'image' and 'angle' keys are present
        and valid; if any error occurs, a minimal fallback result is returned.
        """
        image = problem["image"]
        angle = problem["angle"]

        # Local copy of configuration to avoid attribute lookups
        mode = self.mode
        order = self.order
        reshape = self.reshape

        try:
            rotated = scipy.ndimage.rotate(
                image, angle, reshape=reshape, order=order, mode=mode
            )
        except Exception:
            # In the unlikely event of an error, return an empty array
            return {"rotated_image": np.array([], dtype=image.dtype)}

        return {"rotated_image": rotated}