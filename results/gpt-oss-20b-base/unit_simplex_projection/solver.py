from typing import Any
import numpy as np
import numba as nb

class Solver:

    @staticmethod
    @nb.njit
    def _project(y: np.ndarray) -> np.ndarray:
        """
        Fast Euclidean projection onto the probability simplex.
        Implementation follows the algorithm from the paper
        https://arxiv.org/pdf/1309.1541.
        """
        n = y.size
        # Sort in descending order
        y_sorted = np.sort(y)[::-1]
        # Compute cumulative sums
        cumsum = np.cumsum(y_sorted) - 1.0
        # Find rho: largest i such that y_sorted[i] > cumsum[i]/(i+1)
        rho = 0
        for i in range(n):
            if y_sorted[i] > cumsum[i] / (i + 1):
                rho = i
        theta = cumsum[rho] / (rho + 1)
        # Compute projection
        x = np.empty(n, dtype=y.dtype)
        for i in range(n):
            xi = y[i] - theta
            x[i] = xi if xi > 0 else 0.0
        return x

    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        y = np.asarray(problem.get('y')).flatten()
        x = self._project(y)
        return {'solution': x.tolist()}