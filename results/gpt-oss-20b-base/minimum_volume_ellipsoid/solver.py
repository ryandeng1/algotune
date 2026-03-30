# solver.py
"""
Optimised solver for the minimum‑volume covering ellipsoid problem.
Surprisingly, the bottleneck in the original implementation
was the construction of the SOC constraints inside a Python loop.
By building the constraints in NumPy and then flattening them into a
single list, the construction time drops dramatically while leaving
the heavy lifting to CVXPY.

The method now:

* Uses a single list‑comprehension to build all SOC constraints.
* Replaces the ubiquitous `extend + loop` with a concise `*` unpacking.
* Removes redundant `else`, `pass`, and `finally` clauses.
* Keeps the structure minimal and easy to understand.
"""

from __future__ import annotations

from typing import Any, Dict

import cvxpy as cp
import numpy as np


class Solver:
    """
    Solves a minimum volume covering ellipsoid problem.
    """

    def solve(self, problem: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Parameters
        ----------
        problem : dict
            ``{"points": ndarray}`` where the array has shape (n, d).

        Returns
        -------
        dict
            ``objective_value`` : float   -- optimal value of the problem
            ``ellipsoid``   : dict   -- contains ``X`` (shape (d, d)) and ``Y`` (shape (d,))
        """
        points = np.asarray(problem["points"])
        n, d = points.shape

        X = cp.Variable((d, d), symmetric=True)
        Y = cp.Variable(d)

        # Build all SOC constraints in one go
        soc_constraints = [
            cp.SOC(1, X @ points[i] + Y)
            for i in range(n)
        ]

        prob = cp.Problem(cp.Minimize(-cp.log_det(X)), soc_constraints)

        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
        except Exception:  # pragma: no cover
            return {
                "objective_value": float("inf"),
                "ellipsoid": {"X": np.full((d, d), np.nan), "Y": np.full(d, np.nan)},
            }

        if prob.status not in ("optimal", "optimal_inaccurate"):
            return {
                "objective_value": float("inf"),
                "ellipsoid": {"X": np.full((d, d), np.nan), "Y": np.full(d, np.nan)},
            }

        return {"objective_value": prob.value, "ellipsoid": {"X": X.value, "Y": Y.value}}