import numpy as np
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, np.ndarray]:
        """
        Euclidean projection of y onto the probability simplex.
        Implements the efficient O(n log n) algorithm from
        "Efficient Projections onto the l1-Ball for Learning in High Dimensions"
        by Duchi et al. (2008).
        """
        # Convert the input vector to a 1-D float array
        y = np.asarray(problem.get("y", []), dtype=float).ravel()
        n = y.size

        if n == 0:
            return {"solution": np.array([], dtype=float)}

        # Sort y in descending order
        y_sorted = -np.sort(-y)  # faster than y[::-1] after sort

        # Compute cumulative sum of sorted y
        cum_sum = np.cumsum(y_sorted) - 1.0

        # Find the threshold index rho
        # We need the last index where y_sorted > cum_sum / (1..n)
        # Using broadcasting for the comparison
        right = np.arange(1, n + 1)
        cond = y_sorted > cum_sum / right
        rho = np.nonzero(cond)[0]
        if rho.size == 0:
            # All elements are negative; theta = 0
            theta = 0.0
        else:
            rho = rho[-1]
            theta = cum_sum[rho] / (rho + 1.0)

        # Compute the projected point
        x = np.maximum(y - theta, 0.0)

        return {"solution": x}
