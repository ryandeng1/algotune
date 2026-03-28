import numpy as np
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Solve the Euclidean projection onto the probability simplex.

        This implementation follows the efficient O(n log n) algorithm
        described in https://arxiv.org/pdf/1309.1541.
        """
        y = np.asarray(problem.get("y", []), dtype=np.float64).ravel()
        if y.size == 0:
            return {"solution": []}

        # Compute the sorted vector in decreasing order
        y_sorted = -np.sort(-y)                 # faster descending sort
        cum_y = np.cumsum(y_sorted) - 1

        # Find the threshold index (rho)
        div = np.arange(1, y_sorted.size + 1, dtype=np.float64)
        cond = y_sorted > cum_y / div
        # The last True index gives the correct rho
        rho = np.nonzero(cond)[0][-1]

        theta = cum_y[rho] / (rho + 1.0)
        x = np.maximum(y - theta, 0.0)

        return {"solution": x.tolist()}