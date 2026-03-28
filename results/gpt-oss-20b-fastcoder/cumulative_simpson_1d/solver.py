import numpy as np
from numpy.typing import NDArray

class Solver:
    """
    Optimised solver for the cumulative Simpson integral.
    """

    def solve(self, problem: dict) -> NDArray:
        y: NDArray[np.float64] = problem["y"].astype(np.float64, copy=False)
        dx = float(problem["dx"])

        n = y.size
        if n < 2:
            return np.zeros_like(y)

        # Pre‑allocate result array
        result = np.empty(n, dtype=np.float64)

        # Simpson's constants for cumulative sum
        w = np.empty(n, dtype=np.float64)
        w[0] = 1.0                 # first point weight
        if n > 1:
            # interior points
            w[1:n - 1] = np.where(
                np.arange(1, n - 1) % 2, 4.0, 2.0
            )
            w[-1] = 1.0                # last point weight

        # Cumulative sum of weighted y
        cumw = np.cumsum(w * y)
        result = dx / 3.0 * cumw

        return result