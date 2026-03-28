import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        alpha = np.asarray(problem['alpha'], dtype=float)
        P_total = float(problem['P_total'])
        n = alpha.size

        # Basic feasibility checks
        if n == 0 or P_total <= 0 or not np.all(alpha > 0):
            return {'x': [float('nan')] * n, 'Capacity': float('nan')}

        # Natural bounds for the dual variable lambda
        # For lambda -> 0+, 1/lambda -> +inf
        # For lambda -> inf, 1/lambda -> 0
        lo, hi = 1e-12, np.max(alpha) * 1e8 + 1  # hi large enough

        # Binary search for lambda such that sum max(0, 1/lambda - alpha) = P_total
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            inv = 1.0 / mid
            # compute current sum of x_i
            x = np.maximum(inv - alpha, 0.0)
            s = x.sum()
            if s > P_total:
                lo = mid  # need larger lambda -> smaller 1/lambda
            else:
                hi = mid  # need smaller lambda -> larger 1/lambda
        # Final x using best lambda
        lam = hi
        inv = 1.0 / lam
        x = np.maximum(inv - alpha, 0.0)

        # In rare case numerical errors cause tiny deficit or surplus, rescale
        cur_sum = x.sum()
        if cur_sum > 0:
            x *= P_total / cur_sum

        # Compute capacity
        cap = np.sum(np.log(alpha + x))
        return {'x': x.tolist(), 'Capacity': float(cap)}