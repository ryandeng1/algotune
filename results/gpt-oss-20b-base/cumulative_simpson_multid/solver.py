import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: dict) -> NDArray:
        """
        Compute the cumulative integral along the last axis of the multi‑dimensional array
        using Simpson’s rule.  This implementation avoids the SciPy dependency and is
        fully vectorised for speed.
        """
        y2 = problem["y2"]
        dx = problem["dx"]

        # Ensure data is a float array to avoid accidental integer division
        y2 = np.asarray(y2, dtype=np.float64, order="C")
        n_last = y2.shape[-1]

        # Prepare an output array with the same shape; first element is zero
        result = np.empty_like(y2, dtype=np.float64)
        result[..., 0] = 0.0

        # If there is only one point, integral is zero
        if n_last < 2:
            return result

        # Pre‑compute half step for speed
        two_over_three_dx = (2.0 / 3.0) * dx
        four_over_three_dx = (4.0 / 3.0) * dx
        one_over_three_dx = (1.0 / 3.0) * dx

        # Compute cumulative Simpson integral along the last axis
        # We iterate over the last axis only and use vectorised operations
        for i in range(1, n_last):
            # Simpson contributions for the new slice
            if i % 2 == 1:  # odd index: we start a new pair (i-1, i)
                # Add the integral of a single interval [i-1, i] using the trapezoid rule
                contrib = (y2[..., i - 1] + y2[..., i]) * 0.5 * dx
                result[..., i] = result[..., i - 1] + contrib
            else:  # even index: we complete a pair (i-2, i-1, i)
                # Add the Simpson increment over the last two intervals
                contrib = (
                    two_over_three_dx * y2[..., i - 2]
                    + four_over_three_dx * y2[..., i - 1]
                    + one_over_three_dx * y2[..., i]
                )
                result[..., i] = result[..., i - 1] + contrib

        return result