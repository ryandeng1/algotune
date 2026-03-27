from typing import Any, Dict
import cvxpy as cp
import numpy as np


class Solver:
    def solve(self, problem: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Solve the minimum volume covering ellipsoid (MVCE) problem.

        Parameters
        ----------
        problem : dict
            Must contain key ``"points"``, an ``(n, d)`` array of sample points.

        Returns
        -------
        dict
            ``objective_value`` : optimal (negative log‑determinant) value
            ``ellipsoid`` : dictionary with keys ``X`` (symmetric matrix) and ``Y`` (center)
        """
        points = np.asarray(problem["points"])
        n, d = points.shape

        # Decision variables
        X = cp.Variable((d, d), symmetric=True)
        Y = cp.Variable(d)

        # Constraints:  ||X @ p_i + Y||_2 ≤ 1  for all points
        constraints = []
        for i in range(n):
            constraints.append(cp.norm(X @ points[i] + Y, 2) <= 1)

        # Objective: minimise -logdet(X)
        objective = cp.Minimize(-cp.log_det(X))

        # Problem definition and solution
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
        except Exception:
            # Infeasible / solver error – return empty solution
            return {
                "objective_value": float("inf"),
                "ellipsoid": {"X": np.full((d, d), np.nan), "Y": np.full(d, np.nan)},
            }

        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return {
                "objective_value": float("inf"),
                "ellipsoid": {"X": np.full((d, d), np.nan), "Y": np.full(d, np.nan)},
            }

        return {"objective_value": prob.value, "ellipsoid": {"X": X.value, "Y": Y.value}}