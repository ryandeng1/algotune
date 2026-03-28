import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: dict) -> NDArray:
        """
        Compute the cumulative integral of the 1D array using Simpson's rule
        via a vectorised NumPy implementation for speed.
        """
        y: NDArray = problem["y"]
        dx: float | NDArray = problem["dx"]

        n = y.shape[0]
        if n < 2:
            # Nothing to integrate
            return np.zeros_like(y)

        # Build Simpson weights: 1,4,2,4,2,... with the last element 2 if n is even
        w = np.ones(n, dtype=y.dtype)
        w[1::2] = 4.0
        w[2::2] = 2.0
        w[0] = 1.0
        # Simpson's rule requires an even number of intervals (odd number of points)
        # If n is even, adjust the last weight to 4 for the final interval
        if n % 2 == 0:
            w[-1] = 4.0

        # Compute cumulative weighted sum
        cum_wy = np.cumsum(y * w)
        result = (dx / 3.0) * cum_wy
        # The cumulative integral starts at 0
        result[0] = 0.0
        return result