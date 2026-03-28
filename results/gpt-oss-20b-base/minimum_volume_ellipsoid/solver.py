import numpy as np
import cvxpy as cp

class Solver:
    """
    Solver for the minimum‑volume covering ellipsoid using CVXPY with the ECOS solver.
    """

    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, "Any"]:
        points = np.asarray(problem["points"], dtype=np.float64)
        n, d = points.shape

        # Decision variables
        X = cp.Variable((d, d), symmetric=True)
        Y = cp.Variable(d)

        # SOC constraints: ||X p_i + Y||2 <= 1  for all points
        soc_constraints = [cp.SOC(1, X @ points[i] + Y) for i in range(n)]

        # Objective: maximize log det(X)  <=> minimize -log det(X)
        objective = cp.Minimize(-cp.log_det(X))

        # Problem definition
        prob = cp.Problem(objective, soc_constraints)

        # Solve with the default efficient solver (ECOS)
        prob.solve(solver=cp.ECOS, verbose=False, max_iters=5000)

        # Prepare output
        if prob.status not in ("optimal", "optimal_inaccurate"):
            return {
                "objective_value": float("inf"),
                "ellipsoid": {"X": np.full((d, d), np.nan), "Y": np.full(d, np.nan)},
            }

        return {"objective_value": prob.value, "ellipsoid": {"X": X.value, "Y": Y.value}}