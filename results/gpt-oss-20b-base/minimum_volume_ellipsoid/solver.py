from typing import Any, Dict
import numpy as np
import cvxpy as cp


class Solver:
    """
    Wrapper around a fast CVXPY implementation of the minimum‐volume covering
    ellipsoid.

    The problem is::

          minimize   -log_det(X)            (X ∈ ℝ^d×d, X ⪰ 0)
          subject to ||X p_i + Y||₂ ≤ 1    for all points p_i

    `X` and `Y` are returned; the volume is proportional to
    det(X)^{-1/2}.
    """

    def solve(self, problem: Dict[str, np.ndarray]) -> Dict[str, Any]:
        points = np.asarray(problem["points"], dtype=np.float64)
        n, d = points.shape

        X = cp.Variable((d, d), PSD=True)
        Y = cp.Variable(d)

        # vectorised SOC constraints
        cons = [cp.SOC(1, X @ points[i] + Y) for i in range(n)]

        obj = cp.Minimize(-cp.log_det(X))
        prob = cp.Problem(obj, cons)

        try:
            prob.solve(solver=cp.CLARABEL, warm_start=True, verbose=False)
            if prob.status not in ("optimal", "optimal_inaccurate"):
                return {
                    "objective_value": float("inf"),
                    "ellipsoid": {"X": np.full((d, d), np.nan), "Y": np.full(d, np.nan)},
                }
            return {"objective_value": prob.value, "ellipsoid": {"X": X.value, "Y": Y.value}}
        except Exception:
            return {
                "objective_value": float("inf"),
                "ellipsoid": {"X": np.full((d, d), np.nan), "Y": np.full(d, np.nan)},
            }