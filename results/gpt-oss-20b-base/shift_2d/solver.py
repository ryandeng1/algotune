import numpy as np
from typing import Any, Dict


class Solver:
    def __init__(self, order: int = 0, mode: str = "constant"):
        self.order = order
        self.mode = mode

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Shift a 2D image using NumPy where possible."""
        image: np.ndarray = problem["image"]
        shift_vector = problem["shift"]

        # If the shift is integral and nearest‑neighbour is requested,
        # use the fast NumPy roll function.
        if (
            self.order == 0
            and isinstance(shift_vector, (list, tuple, np.ndarray))
            and all(
                isinstance(v, int) or (isinstance(v, np.integer) and v == int(v))
                for v in shift_vector
            )
        ):
            shifted_image = np.roll(image, shift_vector, axis=(0, 1))
        else:
            # Fallback to generic interpolation using SciPy only if necessary.
            try:
                from scipy.ndimage import shift as ndi_shift

                shifted_image = ndi_shift(
                    image,
                    shift=shift_vector,
                    order=self.order,
                    mode=self.mode,
                )
            except Exception:
                # If SciPy is unavailable or the operation fails,
                # return an empty list to indicate failure.
                return {"shifted_image": []}

        return {"shifted_image": shifted_image}