# solver.py
import numpy as np
from typing import Any, Dict

class Solver:
    """
    Efficient solver for the l1_pruning quadratic program described
    in https://doi.org/10.1109/CVPR.2018.00890.

    The algorithm runs in O(n log n) due to a single sort.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, list]:
        v = np.asarray(problem["v"], dtype=np.float64).ravel()
        k = float(problem["k"])

        # |v| sorted in descending order
        mu = np.sort(np.abs(v), kind="mergesort")[::-1]
        cumsum = np.cumsum(mu)

        # rhs[j] = (cumsum[j] - z) / (j+1)
        divisor = np.arange(1, mu.size + 1, dtype=np.float64)
        rhs = (cumsum - k) / divisor

        # Find first index where mu[j] < rhs[j]
        mask = mu < rhs
        if mask.any():
            idx = np.argmax(mask)  # first True
            theta = rhs[idx]
        else:
            theta = 0.0

        # Compute shrinked vector
        w = np.maximum(np.abs(v) - theta, 0.0) * np.sign(v)

        return {"solution": w.tolist()}