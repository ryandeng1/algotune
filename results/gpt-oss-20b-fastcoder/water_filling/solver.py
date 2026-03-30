# solver.py
# Optimal solution for the problem:
#   maximize   ∑ log(αᵢ + xᵢ)
#   subject to ∑ xᵢ = P_total,    xᵢ ≥ 0
#
# The solution is a closed form water‑filling
#   xᵢ = max(0, 1/λ − αᵢ)
#
# λ is chosen such that the sum constraint holds.
# This implementation is purely numpy based and is
# markedly faster than the original cvxpy formulation.

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np

__all__ = ["Solver"]


class Solver:
    """
    Optimized solver for the logarithmic utility allocation problem.
    """

    @staticmethod
    def _find_lambda(alpha: np.ndarray, P_total: float) -> float:
        """
        Find the Lagrange multiplier λ that satisfies
        Σ max(0, 1/λ − αᵢ) = P_total.
        """
        # Initialise bounds
        # λ must be > 0
        # Upper bound: when λ is very small, (1/λ - αᵢ) → ∞, so sum > P_total
        # Lower bound: when λ is very large, all terms are zero, sum = 0 < P_total
        eps = 1e-12
        lower = eps          # λ → 0⁺ gives huge sum
        # Upper bound: largest α + P_total gives 1/(α+i + P_total) small
        upper = 1 / (np.max(alpha) + P_total)

        # Binary search on λ
        for _ in range(60):  # 60 iterations give ~1e‑18 precision
            mid = (lower + upper) / 2.0
            residual = np.sum(np.maximum(0.0, 1.0 / mid - alpha)) - P_total
            if residual > 0:
                lower = mid
            else:
                upper = mid
        return (lower + upper) / 2.0

    @staticmethod
    def _allocate(alpha: np.ndarray, P_total: float) -> np.ndarray:
        """
        Compute the optimal x vector.
        """
        N = alpha.size
        if N == 0 or P_total <= 0.0 or np.any(alpha <= 0):
            return np.full(N, np.nan, dtype=float)

        lam = Solver._find_lambda(alpha, P_total)
        x = np.maximum(0.0, 1.0 / lam - alpha)

        # Numerical protection: enforce sum constraint exactly
        s = x.sum()
        if s > 1e-15:
            x *= P_total / s
        return x

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parameters
        ----------
        problem : dict
            Must contain keys 'alpha' (array-like) and 'P_total' (float).

        Returns
        -------
        dict
            {'x': list of floats, 'Capacity': float}
        """
        try:
            alpha = np.asarray(problem["alpha"], dtype=float)
            P_total = float(problem["P_total"])
        except (KeyError, ValueError, TypeError):
            return {"x": [], "Capacity": float("nan")}

        x = self._allocate(alpha, P_total)

        if np.any(np.isnan(x)):
            return {"x": [float("nan")] * alpha.size, "Capacity": float("nan")}

        capacity = float(np.sum(np.log(alpha + x)))
        return {"x": x.tolist(), "Capacity": capacity}