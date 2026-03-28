import numpy as np
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List]:
        """
        Solve the sparsity constrained quadratic program:
            min ||x - v||_2^2  subject to  ||x||_1 <= k  and  x = sign(v) * w,
        where w >= 0.  The solution can be obtained in O(n log n).
        """
        # ----- Pre‑processing -------------------------------------------------
        v = np.asarray(problem["v"], dtype=np.float64).ravel()
        k = float(problem["k"])

        # ----- Sub‑problem: compute w from |v| --------------------------------
        u = np.abs(v)
        # Sort descending
        mu = np.sort(u)[::-1]                     # O(n log n)
        # Cumulative sum
        cumsum = np.cumsum(mu)                    # O(n)
        # Potential theta values for every j
        idx = np.arange(1, len(mu) + 1, dtype=np.float64)
        theta_candidates = (cumsum - k) / idx      # element‑wise

        # Find the first j such that mu[j] < theta_candidates[j]
        # Equivalent to: select index j where mu < theta and pick that theta
        # We use vectorized operations; when no such index, use the last one.
        mask = mu < theta_candidates
        if mask.any():
            j = mask.argmax()                     # first True
            theta = theta_candidates[j]
        else:
            theta = theta_candidates[-1]

        # Compute w
        w = np.maximum(u - theta, 0.0)

        # ----- Recover the sign of the original vector ----------------------
        new_v = w * np.sign(v)

        # ----- Return result ------------------------------------------------
        return {"solution": new_v.tolist()}