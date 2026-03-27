import numpy as np
from typing import Any
from numpy.typing import NDArray


class Solver:
    def solve(self, problem: dict) -> NDArray:
        """
        Compute the cumulative integral along the last axis of the multi-dimensional array
        using Simpson's rule.  The implementation is fully vectorised in NumPy and avoids
        the overhead of the SciPy routine.
        """
        y2: NDArray = problem["y2"]
        dx: float = problem["dx"]

        # Simpson's rule coefficients for every trio of points: 1, 4, 2, 4, 2, …
        # We broadcast them over the last axis and use cumulative sums.
        shape_last = y2.shape[-1]
        if shape_last < 2:
            # Not enough points for integration – return zeros of the same shape
            return np.zeros_like(y2)

        # Pre‑compute the weights for every index along the last axis
        # Converted to a 1‑D array so broadcasting is cheap
        ones = np.ones(shape_last, dtype=y2.dtype)
        weights = np.empty(shape_last, dtype=y2.dtype)
        weights[:] = 2  # default
        weights[0] = 1  # first point
        weights[1:] = np.where(
            np.arange(1, shape_last) % 2 == 0, 2, 4
        )
        # Simpson formula scaling factor
        weights *= dx / 3.0

        # Compute the cumulative sum of weighted values along the last axis
        # Use np.cumsum to keep it vectorised; at the first index we want 0
        # We temporarily prepend zeros to achieve that.
        cumsum = np.cumsum(weights * y2, axis=-1)

        # The integral at the first point must be 0
        # Shift cumsum one position to the right and place 0 at start
        result = np.roll(cumsum, shift=1, axis=-1)
        result[..., 0] = 0

        return result