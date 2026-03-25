# solver.py
from __future__ import annotations
from typing import Any, Dict, List
import numpy as np

class Solver:
    @staticmethod
    def _water_filling(alpha: np.ndarray, P_total: float) -> np.ndarray:
        """Analytic water‑filling solution."""
        if P_total <= 0 or np.any(alpha <= 0):
            return np.full_like(alpha, np.nan, dtype=float)

        idx = np.argsort(alpha)          # ascending order
        a_sorted = alpha[idx]
        prefix = np.cumsum(a_sorted)

        n = len(alpha)
        w = None
        for k in range(1, n + 1):
            w_candidate = (P_total + prefix[k - 1]) / k
            if k == n or w_candidate <= a_sorted[k]:
                w = w_candidate
                break

        if w is None:  # fallback
            w = (P_total + prefix[-1]) / n

        x_opt = np.maximum(0.0, w - alpha)

        # Numerical tweak: rescale to exactly match the budget
        total = x_opt.sum()
        if total > 0:
            x_opt *= P_total / total
        return x_opt

    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Solve the water‑filling optimization problem."""
        alpha_list = problem.get("alpha", [])
        P_total = problem.get("P_total", 0.0)

        alpha = np.asarray(alpha_list, dtype=float)
        n = alpha.size

        # Validate input
        if n == 0 or P_total <= 0 or not np.all(alpha > 0):
            return {"x": [float("nan")] * n, "Capacity": float("nan")}

        # Compute optimal allocation using closed form
        x = self._water_filling(alpha, float(P_total))

        # Ensure feasibility (non‑negative, budget fulfilled)
        if np.any(x < 0):
            x = np.maximum(x, 0.0)
        # In rare numerical cases adjust to satisfy the budget
        sum_x = x.sum()
        if sum_x > 0:
            x *= P_total / sum_x

        # Final capacity
        capacity = float(np.sum(np.log(alpha + x)))

        return {"x": x.tolist(), "Capacity": capacity}
