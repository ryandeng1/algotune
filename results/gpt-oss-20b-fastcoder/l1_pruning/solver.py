import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """Fast O(n log n) L1 pruning solution."""
        v = np.asarray(problem["v"], dtype=float).ravel()
        k = problem["k"]

        # Sort absolute values descending and compute cumulative sums
        u = np.abs(v)
        idx = np.argsort(u)[::-1]
        mu = u[idx]
        cumsum = np.cumsum(mu)

        # Find the threshold θ where mu[j] >= (cumsum[j] - k) / (j + 1)
        # Use vectorized search: compute all candidate θ, then pick the first where condition holds
        denom = np.arange(1, len(mu) + 1)
        theta_candidates = (cumsum - k) / denom
        mask = mu >= theta_candidates
        if np.any(mask):
            j = np.argmax(mask)  # first True
            theta = theta_candidates[j]
        else:
            theta = 0.0

        w = np.maximum(u - theta, 0)
        solution = {"solution": (w * np.sign(v)).tolist()}
        return solution