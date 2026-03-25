import numpy as np
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[float]]:
        """
        Project a vector v onto the ℓ1‐ball of radius k, i.e. solve
        min_w ‖v-w‖²  s.t.  ‖w‖₁ ≤ k.

        This implementation follows the classic algorithm by
        Condat (2016) with O(n log n) worst‐case complexity and
        very little overhead.
        """
        v = np.asarray(problem.get("v", []), dtype=np.float64)
        k = float(problem.get("k", 0.0))

        # If the ℓ1 norm is already within the budget, return v unchanged
        if np.abs(v).sum() <= k:
            return {"solution": v.tolist()}

        # Work on absolute values and preserve signs separately
        abs_v = np.abs(v)
        # Sort in descending order
        mu = np.sort(abs_v)[::-1]
        # Compute cumulative sums
        cumsum = np.cumsum(mu)
        # Find the threshold j* such that mu[j] > (cumsum[j]-k)/(j+1)
        # Stop at the first index where the condition fails
        # Using vectorized operations for speed
        rhs = (cumsum - k) / np.arange(1, len(mu) + 1)
        # The threshold index is the last where mu > rhs
        idx = np.where(mu > rhs)[0]
        j_star = idx[-1] if idx.size > 0 else -1

        theta = (cumsum[j_star] - k) / (j_star + 1) if j_star >= 0 else 0.0
        # Shrinkage
        w = np.maximum(abs_v - theta, 0.0)
        # Restore signs
        w = w * np.sign(v)

        return {"solution": w.tolist()}
