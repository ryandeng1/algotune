import numpy as np
import scipy.ndimage
from typing import Any, Dict

class Solver:
    """Efficient 2‑D image shift using SciPy's ndimage.shift."""

    # cubic spline interpolation order and constant padding with 0
    order: int = 3
    mode: str = "constant"

    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Shift a 2‑D image by a sub‑pixel amount using cubic spline interpolation
        with constant padding.

        Parameters
        ----------
        problem : dict
            Must contain:
                'image' : 2‑D array‑like of floats.
                'shift' : list or array of two floats [shift_row, shift_col].

        Returns
        -------
        dict
            {'shifted_image': np.ndarray of same shape as input image}
        """
        # Extract inputs
        image = np.asarray(problem["image"], dtype=float)
        shift_vec = np.asarray(problem["shift"], dtype=float)

        # Perform shift
        shifted = scipy.ndimage.shift(
            image,
            shift=shift_vec,
            order=self.order,
            mode=self.mode,
            cval=0.0,
        )

        return {"shifted_image": shifted}
