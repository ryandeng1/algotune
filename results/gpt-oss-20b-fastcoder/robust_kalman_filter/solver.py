import cvxpy as cp
import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        A = np.asarray(problem["A"], dtype=float)
        B = np.asarray(problem["B"], dtype=float)
        C = np.asarray(problem["C"], dtype=float)
        y = np.asarray(problem["y"], dtype=float)
        x0 = np.asarray(problem["x_initial"], dtype=float)
        tau = float(problem["tau"])
        M = float(problem["M"])

        N, m = y.shape
        n = A.shape[1]
        p = B.shape[1]

        # Variables
        x = cp.Variable((N + 1, n))
        w = cp.Variable((N, p))
        v = cp.Variable((N, m))

        # Objective
        obj = cp.Minimize(cp.sum_squares(w) + tau * cp.sum(cp.huber(cp.norm(v, axis=1), M)))

        # Constraints
        constraints = [x[0] == x0]
        for t in range(N):
            constraints.append(x[t + 1] == A @ x[t] + B @ w[t])
            constraints.append(y[t] == C @ x[t] + v[t])

        prob = cp.Problem(obj, constraints)
        try:
            prob.solve(solver=cp.ECOS, warm_start=True, verbose=False)
        except Exception:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x.value is None:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        return {
            "x_hat": x.value.tolist(),
            "w_hat": w.value.tolist(),
            "v_hat": v.value.tolist()
        }