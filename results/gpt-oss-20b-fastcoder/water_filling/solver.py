from typing import Any, Dict, List
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        alpha_raw = problem.get('alpha')
        P_total_raw = problem.get('P_total')

        try:
            alpha = np.asarray(alpha_raw, dtype=float)
            P_total = float(P_total_raw)
        except Exception:
            return {'x': [float('nan')] * (len(alpha_raw) if alpha_raw else 0),
                    'Capacity': float('nan')}

        n = alpha.size
        if n == 0 or P_total <= 0 or not np.all(alpha > 0):
            return {'x': [float('nan')] * n, 'Capacity': float('nan')}

        # Water‑filling via Lagrange multiplier
        idx = np.argsort(alpha)
        a_sorted = alpha[idx]
        cum_sum = np.cumsum(a_sorted)

        k = None
        for i in range(n):
            tot = P_total + cum_sum[i]
            lam = 1.0 / (tot / (i + 1))   # candidate lambda
            if lam <= 1.0 / a_sorted[i]:
                k = i + 1
                break
        if k is None:
            k = n
            lam = 1.0 / ((P_total + cum_sum[n-1]) / n)

        x_sorted = np.zeros(n)
        valid_mask = a_sorted < 1.0 / lam
        x_sorted[valid_mask] = 1.0 / lam - a_sorted[valid_mask]

        # Convert back to original order
        x = np.empty(n, dtype=float)
        x[idx] = x_sorted

        # Numerical safety
        if np.any(x < 0):
            x = np.maximum(x, 0.0)
        current_sum = x.sum()
        if current_sum > 1e-12:
            x *= P_total / current_sum

        capacity = np.sum(np.log(alpha + x))
        if not np.isfinite(capacity):
            capacity = float('nan')

        return {'x': x.tolist(), 'Capacity': float(capacity)}