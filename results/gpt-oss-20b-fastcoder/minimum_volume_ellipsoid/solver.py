import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, any]:
        points = np.array(problem["points"])
        n, d = points.shape
        X = cp.Variable((d, d), symmetric=True)
        Y = cp.Variable(d)
        constraints = [cp.SOC(1, X @ points[i] + Y) for i in range(n)]
        objective = cp.Minimize(-cp.log_det(X))
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
            if prob.status not in ["optimal", "optimal_inaccurate"]:
                return {
                    "objective_value": float("inf"),
                    "ellipsoid": {"X": np.full((d, d), np.nan), "Y": np.full(d, np.nan)},
                }
            return {
                "objective_value": prob.value,
                "ellipsoid": {"X": X.value, "Y": Y.value},
            }
        except Exception:
            return {
                "objective_value": float("inf"),
                "ellipsoid": {"X": np.full((d, d), np.nan), "Y": np.full(d, np.nan)},
            }