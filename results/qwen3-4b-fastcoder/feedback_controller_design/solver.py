from typing import Any
import cvxpy as cp
import numpy as np


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        A = np.array(problem["A"])
        B = np.array(problem["B"])
        n, m = A.shape[0], B.shape[1]

        Q = cp.Variable((n, n), symmetric=True)
        L = cp.Variable((m, n))
        constraints = [
            cp.bmat([[Q, Q @ A.T + L.T @ B.T], [A @ Q + B @ L, Q]]) >> np.eye(2 * n),
            Q >> np.eye(n),
        ]
        objective = cp.Minimize(0)
        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.CLARABEL)
            if prob.status in ["optimal", "optimal_inaccurate"]:
                inv_Q = np.linalg.inv(Q.value)
                K_value = L.value @ inv_Q
                P = inv_Q
                return {"is_stabilizable": True, "K": K_value.tolist(), "P": P.tolist()}
            else:
                return {"is_stabilizable": False, "K": None, "P": None}
        except Exception as e:
            return {"is_stabilizable": False, "K": None, "P": None}