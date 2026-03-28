import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: dict) -> NDArray:
        """
        Compute the cumulative integral of a 1‑D array using Simpson's rule
        but without calling the Python‑level `scipy.integrate.cumulative_simpson`.
        This implementation uses fully vectorised NumPy operations for maximum
        speed.
        """
        y = problem["y"]
        dx = problem["dx"]

        # Force a 1-D float array
        y = np.asarray(y, dtype=np.float64).ravel()
        n = y.size

        if n == 0:
            return y.copy()

        # If number of points is even, drop last point to make it odd
        # (Simpson's rule requires an odd number of points).
        if (n - 1) % 2 == 0:   # n-1 is even => n is odd => ok
            pass
        else:
            y = y[:-1]
            n = y.size

        # Compute Simpson weights for all points
        # w[0] = 1, w[-1] = 1, w[odd indices] = 4, w[even indices] = 2
        weights = np.empty(n, dtype=np.float64)
        weights[0] = 1.0
        weights[-1] = 1.0
        if n > 2:
            weights[1:-1:2] = 4.0
            weights[2:-1:2] = 2.0

        # Cumulative sum of weighted values
        weighted = y * weights
        cumsum = np.cumsum(weighted)

        # Multiply by dx/3
        result = cumsum * (dx / 3.0)
        return result