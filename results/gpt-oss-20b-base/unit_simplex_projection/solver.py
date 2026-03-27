from typing import Any
import numpy as np


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Compute the projection of vector `y` onto the probability simplex
        using the O(n log n) algorithm based on sorting.
        """
        y = np.asanyarray(problem["y"], dtype=float).ravel()
        n = y.size

        # Sort in decreasing order
        sorted_y = np.sort(y)[::-1]

        # Cumulative sum minus 1
        cumsum = np.cumsum(sorted_y) - 1.0

        # Find rho: the largest index where sorted_y > cumsum / (k+1)
        k = np.arange(1, n + 1)
        cond = sorted_y > cumsum / k
        rho = cond.nonzero()[0][-1]  # guaranteed at least one True

        theta = cumsum[rho] / (rho + 1.0)

        # Compute the projected vector
        x = np.maximum(y - theta, 0.0)
        return {"solution": x.tolist()}