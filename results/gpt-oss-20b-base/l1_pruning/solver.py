# solver.py
import numpy as np
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[float]]:
        """
        Project the vector v onto the L1 ball of radius k.
        This implementation follows the classic algorithm by Duchi et al.
        (Efficient Projections onto the L1-Ball for Learning in High Dimensions,
          ICML 2008). The time complexity is O(n log n) due to sorting.
        """
        v = np.asarray(problem["v"], dtype=np.float64)
        k = problem["k"]

        # If the L1 norm is already ≤ k, return the original vector
        if np.sum(np.abs(v)) <= k:
            return {"solution": v.tolist()}

        # Sort the absolute values in descending order
        abs_v = np.abs(v)
        sorted_abs = np.sort(abs_v)[::-1]
        cum_sum = np.cumsum(sorted_abs)

        # Find the threshold rho
        inds = np.arange(1, len(sorted_abs) + 1)
        # Condition: sorted_abs[i] - (cum_sum[i-1] - k) / i > 0
        # Equivalent to finding the largest i where sorted_abs[i-1] > (cum_sum[i-1] - k) / i
        theta_candidates = (cum_sum - k) / inds
        mask = sorted_abs > theta_candidates
        if not np.any(mask):
            rho = 0
            theta = 0.0
        else:
            rho = np.max(np.where(mask)[0])  # zero‑based index
            theta = theta_candidates[rho]

        # Soft–thresholding
        w = np.sign(v) * np.maximum(abs_v - theta, 0.0)
        return {"solution": w.tolist()}
