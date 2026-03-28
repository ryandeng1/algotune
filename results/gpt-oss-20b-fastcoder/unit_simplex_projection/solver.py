import numpy as np
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, list]:
        """
        Projection of y onto probability simplex in O(n log n) via sorting.
        """
        # Convert to 1‑D numpy array
        y = np.asarray(problem["y"]).flat
        n = y.size

        # Sort descending
        sorted_y = np.sort(y)[::-1]      # shape (n,)
        # Cumulative sum of sorted values, subtract 1
        cumsum_y = np.cumsum(sorted_y) - 1.0

        # Find rho = max{ j | sorted_y[j] > cumsum_y[j]/(j+1) }
        pos = np.arange(1, n + 1, dtype=float)
        cond = sorted_y > cumsum_y / pos
        if not cond.any():
            rho = 0
        else:
            rho = np.nonzero(cond)[0][-1]

        # Threshold
        theta = cumsum_y[rho] / (rho + 1.0)

        # Projection
        x = np.maximum(y - theta, 0.0)
        return {"solution": x.tolist()}