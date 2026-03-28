import numpy as np
import cvxpy as cp
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Solves a minimum volume covering ellipsoid problem using CVXPY.

        Parameters
        ----------
        problem : dict
            Dictionary containing a single key 'points', which is a 2‑D numpy array
            of shape (n, d) with the points to be enclosed.

        Returns
        -------
        dict
            Dictionary with keys:
                - objective_value: the optimal objective value (≈ log(volume)).
                - ellipsoid: a dict with the optimal symmetric matrix X and vector Y.
        """
        points = np.asarray(problem["points"], dtype=np.float64)
        n, d = points.shape

        X = cp.Variable((d, d), symmetric=True)
        Y = cp.Variable(d)

        constraints = [cp.SOC(1, X @ points[i] + Y) for i in range(n)]
        prob = cp.Problem(cp.Minimize(-cp.log_det(X)), constraints)

        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
        except Exception:
            return {
                "objective_value": float("inf"),
                "ellipsoid": {"X": np.full((d, d), np.nan), "Y": np.full(d, np.nan)},
            }

        if prob.status not in {"optimal", "optimal_inaccurate"}:
            return {
                "objective_value": float("inf"),
                "ellipsoid": {"X": np.full((d, d), np.nan), "Y": np.full(d, np.nan)},
            }

        return {
            "objective_value": prob.value,
            "ellipsoid": {"X": X.value, "Y": Y.value},
        }