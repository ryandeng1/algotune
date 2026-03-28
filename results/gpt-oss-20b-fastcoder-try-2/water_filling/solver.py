import numpy as np
from typing import Any

class Solver:
    """
    Fast analytical solver for:
        maximize sum_i log(alpha_i + x_i)
        s.t.   sum_i x_i = P_total,   x_i >= 0, alpha_i > 0
    """
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Input extraction and validation
        alpha = np.asarray(problem["alpha"], dtype=np.float64)
        n = alpha.size
        try:
            P_total = float(problem["P_total"])
        except (KeyError, ValueError, TypeError):
            return {"x": [float("nan")] * n, "Capacity": float("nan")}

        # Basic checks
        if n == 0 or P_total <= 0 or not np.all(alpha > 0):
            return {"x": [float("nan")] * n, "Capacity": float("nan")}

        # Sort alphas ascending for efficient threshold search
        idx = np.argsort(alpha)
        alpha_sorted = alpha[idx]

        # Compute cumulative sums of alpha
        cum_alpha = np.cumsum(alpha_sorted)

        # Helper: compute Lagrange multiplier lambda for a given k
        def lambda_for_k(k: int) -> float:
            # (k items participate) => (P_total + sum_{i=1..k} alpha_i) / k
            return (P_total + cum_alpha[k-1]) / k

        # Find largest k such that x_i >= 0 → 1/lambda > alpha_i
        # Equivalent to lambda > 1/(alpha_i + eps). We binary-search over k.
        low, high = 1, n
        best_k = 0
        while low <= high:
            mid = (low + high) // 2
            lam = lambda_for_k(mid)
            if lam > alpha_sorted[mid - 1]:
                best_k = mid
                low = mid + 1
            else:
                high = mid - 1

        # Compute final x using best_k
        lam = lambda_for_k(best_k)
        x_part = 1.0 / lam - alpha_sorted[:best_k]
        x = np.zeros(n, dtype=np.float64)
        x[:best_k] = x_part
        # Remaining x_i are zero
        # Reorder to original order
        x_ordered = np.empty_like(x)
        x_ordered[idx] = x

        # Compute capacity
        final_terms = np.log(alpha + x_ordered)
        capacity = float(np.sum(final_terms))

        return {"x": x_ordered.tolist(), "Capacity": capacity}