import numpy as np
from numpy.typing import NDArray


class Solver:
    def solve(self, problem: dict) -> NDArray:
        """
        Compute the cumulative integral of a 1‑D array using Simpson's rule
        but without the overhead of SciPy's `cumulative_simpson`.
        """
        y: NDArray = problem["y"]
        dx: float | NDArray = problem["dx"]

        # Ensure we work with 1‑D array of floats
        y = np.asarray(y, dtype=float).ravel()
        n = y.size

        if n < 2:
            return np.empty_like(y)

        # For internal points the Simpson weights are 4, 2, 4, 2, ...
        # Create the weight array: first element 1, last element 1
        w = np.empty(n, dtype=float)
        w[0] = 1.0
        w[-1] = 1.0
        if n > 2:
            w[1:-1] = np.where(np.arange(1, n-1) % 2, 4.0, 2.0)

        # Weighted cumulative sum
        cum = np.cumsum(y * w) * (dx / 3.0)
        return cum