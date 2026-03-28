from typing import Any
import cvxpy as cp
import numpy as np


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        A = np.array(problem["A"])
        n = A.shape[0]
        P = cp.Variable((n, n), symmetric=True)
        constraints = [P - A.T @ P @ A - np.eye(n) >> 0]
        prob = cp.Problem(cp.Minimize(0), constraints)

        try:
            prob.solve(solver=cp.CLARABEL, verbose=False, eps=1e-8)
            if prob.status in ["optimal", "optimal_inaccurate"]:
                return {"is_stable": True, "P": P.value.tolist()}
            else:
                return {"is_stable": False, "P": None}

        except Exception as e:
            return {"is_stable": False, "P": None}