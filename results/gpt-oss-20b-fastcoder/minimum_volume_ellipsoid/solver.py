"""
solver.py

Implementation of a fast solver for the minimum volume covering ellipsoid
problem.  The implementation aims to keep the CVXPY modeling phase
minimal and to trigger the solver as efficiently as possible.

The problem is formulated as:

    minimize   -log det(X)
    subject to [ 1           (X*p_i + Y)^T  ]
                [ (X*p_i + Y)   X          ]  ⪰ 0   for all points p_i

where X is a symmetric d×d matrix, Y is a d‑dimensional centre vector
and each point p_i ∈ ℝ^d is guaranteed to lie inside the ellipsoid.
"""

from __future__ import annotations

from typing import Any, Dict

import cvxpy as cp
import numpy as np


class Solver:
    """
    Solver for the minimum volume covering ellipsoid problem.

    Attributes
    ----------
    None
    """

    def solve(self, problem: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Solve the minimum volume covering ellipsoid problem.

        Parameters
        ----------
        problem
            Dictionary containing a single key ``'points'``.
            The value is a NumPy array of shape (N, d) holding the
            points that must be contained in the ellipsoid.

        Returns
        -------
        dict
            Dictionary with keys:

            * ``objective_value`` : float, optimal value of the convex
              objective (negative log‑determinant of X).  The actual volume
              is `exp(-objective_value/d)`.
            * ``ellipsoid`` : dict with symmetric matrix ``X`` and centre
              vector ``Y`` that describe the optimal ellipsoid.
              If the problem is infeasible or the solver fails,
              NaNs are returned.
        """
        points = np.asarray(problem["points"], dtype=np.float64, order="C")
        if points.ndim != 2:
            raise ValueError("`points` must be a 2‑D array of shape (N, d).")
        N, d = points.shape

        # Decision variables
        X = cp.Variable((d, d), PSD=True)  # X must be positive semidefinite
        Y = cp.Variable(d)

        # Build constraints in a vectorised manner:
        #   (X*p_i + Y) is a vector of shape (d,)
        #   We need SOC: (1, X*p_i + Y) ⪰ 0 for each i.
        # Use a list comprehension – CVXPY incants this internally.
        cons = [cp.SOC(1, X @ points[i] + Y) for i in range(N)]

        # Objective: minimise -log(det(X))
        objective = cp.Minimize(-cp.log_det(X))

        # Problem definition
        prob = cp.Problem(objective, cons)

        # Solve
        try:
            # ECOS is usually fast for this problem size and does not need
            # optional warm starts.  We set verbose=False to keep output clean.
            prob.solve(solver=cp.ECOS, verbose=False, warm_start=True)

            if prob.status not in {"optimal", "optimal_inaccurate"}:
                # Infeasible / solver failure
                return {
                    "objective_value": float("inf"),
                    "ellipsoid": {"X": np.full((d, d), np.nan), "Y": np.full(d, np.nan)},
                }

            return {
                "objective_value": float(prob.value),
                "ellipsoid": {"X": X.value, "Y": Y.value},
            }

        except Exception:  # pragma: no cover
            # Any exception indicates we cannot recover the solution.
            return {
                "objective_value": float("inf"),
                "ellipsoid": {"X": np.full((d, d), np.nan), "Y": np.full(d, np.nan)},
            }