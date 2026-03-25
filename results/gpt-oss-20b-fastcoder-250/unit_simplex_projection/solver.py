# solver.py
import numpy as np
from typing import Any, Dict


class Solver:
    """
    Euclidean projection of a point onto the probability simplex.
    Implements the O(n log n) algorithm from
    Wang & Carreira-Perpinán 2013.
    """
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        y = np.asarray(problem.get("y", []), dtype=np.float64).flatten()
        n = y.size
        if n == 0:
            return {"solution": np.array([], dtype=np.float64)}

        # Sort y in descending order
        sorted_y = np.sort(y)[::-1]

        # Compute cumulative sum of sorted_y minus 1
        cumsum = np.cumsum(sorted_y)
        # Compute threshold condition
        rho = np.nonzero(sorted_y > (cumsum - 1) / np.arange(1, n + 1))[0]
        rho = rho[-1]  # last index where condition holds

        theta = (cumsum[rho] - 1) / (rho + 1)
        x = np.maximum(y - theta, 0.0)
        return {"solution": x}
