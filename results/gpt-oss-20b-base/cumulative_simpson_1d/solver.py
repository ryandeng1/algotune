import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: dict, **kwargs) -> NDArray:
        """
        Compute the cumulative integral of a 1D array using Simpson's rule.
        This implementation is a lightweight replacement for scipy's
        cumulative_simpson that works for evenly spaced data.
        """
        y = np.asarray(problem["y"], dtype=np.float64)
        dx = float(problem["dx"])
        n = y.size

        # Allocate result array
        result = np.empty(n, dtype=np.float64)
        result[0] = 0.0

        if n == 1:
            return result

        # First interval by trapezoidal rule
        result[1] = 0.5 * dx * (y[0] + y[1])

        # Loop over remaining points
        for i in range(2, n):
            if (i % 2) == 0:
                # Apply Simpson's rule over the last pair of intervals
                result[i] = result[i - 2] + dx / 3.0 * (
                    y[i - 2] + 4.0 * y[i - 1] + y[i]
                )
            else:
                # Trapezoidal for the single interval
                result[i] = result[i - 1] + 0.5 * dx * (y[i - 1] + y[i])

        return result
