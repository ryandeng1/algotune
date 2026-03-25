import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, Any]:
        points = np.array(problem["points"])
        n, d = points.shape
        X = cp.Variable((d, d), symmetric=True)
        Y = cp.Variable((d,))
        constraints = [cp.SOC(1, X @ points[i] + Y) for i in range(n)]
        problem = cp.Problem(cp.Minimize(-cp.log_det(X)), constraints)
        try:
            problem.solve(solver=cp.CLARABEL, verbose=False)
            if problem.status not in ["optimal", "optimal_inaccurate"]:
                return {
                    "objective_value": float("inf"),
                    "ellipsoid": {"X": np.nan * np.ones((d, d)), "Y": np.nan * np.ones((d,))},
                }
            return {
                "objective_value": problem.value,
                "ellipsoid": {"X": X.value, "Y": Y.value}
            }
        except Exception as e:
            return {
                "objective_value": float("inf"),
                "ellipsoid": {"X": np.nan * np.ones((d, d)), "Y": np.nan * np.ones((d,))},
            }
