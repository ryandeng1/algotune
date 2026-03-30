# solver.py
from typing import Any, Dict

import numpy as np


class Solver:
    """
    Optimises a concave problem::

        maximise     ∑ log(alphaᵢ + xᵢ)
        subject to   ∑ xᵢ = P_total
                     xᵢ ≥ 0

    The KKT conditions yield a closed‑form solution that can be found by a
    simple 1‑D search on the Lagrange multiplier λ.
    """

    @staticmethod
    def _solve(alpha: np.ndarray, P_total: float) -> np.ndarray:
        """
        Return the optimal x vector for given alpha and P_total.
        Assumes all inputs are positive (validation performed in solve()).
        """
        # Helper to compute remaining power for a given λ
        def remaining(power: float, lam: float, a: np.ndarray) -> float:
            # xᵢ = 1/λ - aᵢ if positive
            # -> contribution of xᵢ to sum(x) is 1/λ - aᵢ
            # total sum is sum(max(0, 1/λ - aᵢ))
            inv_lam = 1.0 / lam
            contrib = inv_lam - a
            contrib[contrib < 0] = 0.0
            return np.sum(contrib)

        # Validate: in case all alphas are huge, we might not use any power
        # Then the solution is simply any feasible x, e.g. zeros.
        n = alpha.size
        if P_total <= 0 or n == 0:
            return np.full(n, np.nan)

        # Initial bounds for λ:
        # If we had to use all power, λ would be small (xᵢ large).
        # Minimal λ is 1/(max(alpha)+epsilon) to avoid division by zero.
        lam_min = 1.0 / (alpha.max() + 1e-12)
        # Max λ gives zero allocation
        lam_max = 1000.0 + alpha.max()  # large enough

        # Binary search for λ such that remaining power equals P_total
        for _ in range(60):  # 60 iterations gives sub‑micro precision
            lam_mid = (lam_min + lam_max) / 2.0
            used = remaining(P_total, lam_mid, alpha)
            if used > P_total:
                lam_min = lam_mid
            else:
                lam_max = lam_mid

        lam_opt = (lam_min + lam_max) / 2.0
        inv_lam = 1.0 / lam_opt
        x = inv_lam - alpha
        x[x < 0] = 0.0
        return x

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        try:
            alpha = np.asarray(problem['alpha'], dtype=float)
            P_total = float(problem['P_total'])
        except Exception:
            n = problem.get('n', 0)
            return {'x': [np.nan] * n, 'Capacity': np.nan}

        alpha = alpha.ravel()
        if (alpha.size == 0 or P_total <= 0
                or not np.all(alpha > 0)):
            return {'x': [np.nan] * alpha.size, 'Capacity': np.nan}

        x = self._solve(alpha, P_total)

        if np.isnan(x).any():
            return {'x': [np.nan] * alpha.size, 'Capacity': np.nan}

        # Ensure strict feasibility (total sum ≈ P_total)
        total = x.sum()
        if total > 0:
            scale = P_total / total
            if not np.isclose(scale, 1.0):
                x *= scale

        capacity = float(np.sum(np.log(alpha + x)))
        if not np.isfinite(capacity):
            capacity = np.nan

        return {'x': x.tolist(), 'Capacity': capacity}