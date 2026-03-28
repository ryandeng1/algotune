from typing import Any, Dict, List
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Optimized solver for the L1–pruning problem.
        """
        # Extract data
        v = np.asarray(problem["v"], dtype=np.float64).ravel()
        k = float(problem["k"])

        # Absolute values and sorting (descending)
        u = np.abs(v)
        mu = np.sort(u, kind="mergesort")[::-1]
        n = mu.size

        # Cumulative sums and candidate thresholds
        cumsum = np.cumsum(mu)
        idx = np.arange(1, n + 1)
        theta_candidates = (cumsum - k) / idx

        # Find first index where mu < theta_candidate
        mask = mu < theta_candidates
        if mask.any():
            theta = theta_candidates[mask.argmax()]
        else:
            theta = 0.0

        # Compute solution vector
        w = np.maximum(u - theta, 0.0)
        pruned = w * np.sign(v)

        return {"solution": pruned.tolist()}