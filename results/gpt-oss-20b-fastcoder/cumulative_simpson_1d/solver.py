import numpy as np
from numpy.typing import NDArray

class Solver:
    """Fast cumulative Simpson integration without external dependencies."""

    def solve(self, problem: dict) -> NDArray:
        """
        Compute the cumulative integral of a 1‑D array using Simpson's rule.

        Parameters
        ----------
        problem : dict
            Dictionary with keys 'y' (array of function values) and
            'dx' (spacing).

        Returns
        -------
        NDArray
            Cumulative integral values at each index.
        """
        y = np.asarray(problem["y"], dtype=np.float64)
        dx = float(problem["dx"])
        n = y.size

        # Result array
        out = np.empty(n, dtype=np.float64)
        if n == 0:
            return out

        out[0] = 0.0  # integral from 0 to the first point is zero

        # For 1 element, integral is zero as above.
        if n == 1:
            return out

        # Start with the first Simpson segment (indices 0,1,2)
        if n >= 3:
            out[2] = dx * (y[0] + 4.0 * y[1] + y[2]) / 6.0
            start = 2
        else:  # n == 2, use trapezoidal rule as last segment incomplete
            out[1] = dx * (y[0] + y[1]) / 2.0
            start = 1

        # Continue with remaining Simpson segments
        for i in range(start + 2, n, 2):
            # Simpson rule for segment (i-2, i-1, i)
            segment = dx * (y[i - 2] + 4.0 * y[i - 1] + y[i]) / 6.0
            out[i] = out[i - 2] + segment

        # Handle any trailing point (odd length) with a trapezoidal segment
        if n % 2 == 0 and n > 2:
            last = n - 1
            out[last] = out[last - 1] + dx * (y[last - 1] + y[last]) / 2.0
        elif n == 2:
            # already handled
            pass

        return out