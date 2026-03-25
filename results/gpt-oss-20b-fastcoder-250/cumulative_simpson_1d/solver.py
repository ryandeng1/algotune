import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: dict, **kwargs) -> NDArray:
        """
        Compute the cumulative integral of a 1‑D array using Simpson's rule.
        Implements the algorithm in a fully vectorised manner for speed.
        """
        y: NDArray = np.asarray(problem["y"], dtype=np.float64)
        dx: float = float(problem["dx"])

        n = y.size
        if n < 2:
            return np.zeros(n, dtype=float)

        # Simpson incremental areas for segments starting at each i-2
        inc = dx / 3.0 * (y[:-2] + 4.0 * y[1:-1] + y[2:])

        # cumulative sum of incremental areas, with zeros for the first two points
        result = np.concatenate(([0.0, 0.0], np.cumsum(inc)))
        return result
