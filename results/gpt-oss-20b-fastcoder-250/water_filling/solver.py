# solver.py
from __future__ import annotations

import numpy as np
from typing import Any, Dict

class Solver:
    """
    Fast closed‑form solver for the classic water‑filling concave optimisation
    problem.
    """
    @staticmethod
    def _water_filling(alpha: np.ndarray, P_total: float) -> np.ndarray:
        """
        Compute optimal power allocation x* for given channel gains `alpha`
        and total power budget `P_total`.

        Returns a numpy array of shape (n,) with the optimal x values.
        """
        # Sort alphas ascending and keep mapping
        idx = np.argsort(alpha)
        a_sorted = alpha[idx]
        n = a_sorted.size

        # Cumulative sums of sorted alphas
        prefix = np.cumsum(a_sorted)

        # Find water level w
        w = None
        for k in range(1, n + 1):
            # candidate water level if first k channels are active
            w_candidate = (P_total + prefix[k - 1]) / k
            if k == n or w_candidate <= a_sorted[k]:
                w = w_candidate
                break
        if w is None:  # fallback
            w = (P_total + prefix[-1]) / n

        # Unsorted solution
        x_unscaled = np.maximum(0.0, w - alpha)

        # Numerical scaling to meet the exact budget
        total = x_unscaled.sum()
        if total > 0:
            scale = P_total / total
            x_unscaled *= scale
        return x_unscaled

    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Compute the water‑filling allocation and capacity.

        Parameters
        ----------
        problem : dict
            Must contain keys 'alpha' (list/array of positives) and
            'P_total' (positive float).

        Returns
        -------
        dict
            {'x': list of allocations, 'Capacity': total capacity}
        """
        # Extract inputs
        alpha = np.asarray(problem.get("alpha"), dtype=np.float64)
        P_total = float(problem.get("P_total", np.nan))

        n = alpha.size

        # Validate basic conditions
        if n == 0 or P_total <= 0.0 or not np.all(alpha > 0.0):
            return {"x": [float("nan")] * n, "Capacity": float("nan")}

        # Compute optimal allocation
        x_opt = self._water_filling(alpha, P_total)

        # Compute capacity
        capacity = float(np.sum(np.log(alpha + x_opt)))

        return {"x": x_opt.tolist(), "Capacity": capacity}
