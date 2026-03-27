from typing import Any
import cvxpy as cp
import numpy as np


class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, Any]:
        """
        Solves a minimum volume covering ellipsoid problem using CVXPY.

        The formulation follows:
        minimize   -log_det(X)
        subject to X ≽ 0
                   || X (p_i - Y) ||₂ ≤ 1    ∀ i

        Parameters
        ----------
        problem
            Dictionary with key ``points`` containing an (n, d)-array of points.

        Returns
        -------
        dict
            ``objective_value``: optimal objective (proportional to log(ellipsoid volume)).
            ``ellipsoid``: dictionary with symmetric matrix ``X`` and centre ``Y``.
        """
        # Extract points
        points = np.array(problem["points"], dtype=np.float64)
        n, d = points.shape

        # Decision variables
        X = cp.Variable((d, d), symmetric=True)
        Y = cp.Variable(d)

        # Build constraints in a vectorized way
        # Inequality: || X @ (p_i - Y) ||₂ ≤ 1  ⇔  SOC(1, X @ (p_i - Y))
        diff = points - Y  # shape (n, d), broadcasted over Y
        # Since X is a matrix variable, use a custom expression:
        # we create an array of linear expressions of shape (n, d)
        # Each row: X @ (p_i - Y)
        expr = [X @ (points[i] - Y) for i in range(n)]
        constraints = [cp.SOC(1, ex) for ex in expr]

        # Problem definition
        prob = cp.Problem(cp.Minimize(-cp.log_det(X)), constraints)

        # Solve with CLARABEL (or default if not available)
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
        except Exception:
            prob.solve(verbose=False)

        # Prepare result
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            return {
                "objective_value": float("inf"),
                "ellipsoid": {"X": np.full((d, d), np.nan), "Y": np.full(d, np.nan)},
            }

        return {
            "objective_value": float(prob.value),
            "ellipsoid": {"X": X.value, "Y": Y.value},
        }