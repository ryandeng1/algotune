from typing import Any
import cvxpy as cp
import numpy as np


class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, Any]:
        c = problem["c"]
        b = problem["b"]
        P = problem["P"]
        q = problem["q"]
        m = len(P)
        n = len(c)

        x = cp.Variable(n)

        constraint = [cp.SOC(b[i] - q[i].T @ x, P[i].T @ x) for i in range(m)]

        problem = cp.Problem(cp.Minimize(c.T @ x), constraint)

        try:
            problem.solve(solver=cp.CLARABEL, verbose=False)

            if problem.status not in ["optimal", "optimal_inaccurate"]:
                return {"objective_value": float("inf"), "x": np.array([np.nan] * n)}

            return {"objective_value": problem.value, "x": x.value}

        except Exception:
            return {"objective_value": float("inf"), "x": np.array([np.nan] * n)}