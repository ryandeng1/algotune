# solver.py
from typing import Any, Dict, List

import numpy as np
import cvxpy as cp


class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Robust Kalman filter using a Huber loss.

        The formulation is a standard convex quadratic program, therefore we
        use CVXPY to construct and solve it.  OSQP is chosen as the solver
        because it handles large quadratic problems very efficiently.
        """
        # Extract data
        A = np.asarray(problem["A"])
        B = np.asarray(problem["B"])
        C = np.asarray(problem["C"])
        y = np.asarray(problem["y"])
        x0 = np.asarray(problem["x_initial"])
        tau = float(problem["tau"])
        M = float(problem["M"])

        N, m = y.shape
        n = A.shape[1]
        p = B.shape[1]

        # Decision variables
        x = cp.Variable((N + 1, n), name="x")
        w = cp.Variable((N, p), name="w")
        v = cp.Variable((N, m), name="v")

        # Objective
        # quadratic part in process noise
        process_term = cp.sum_squares(w)
        # Huber loss on measurement noise
        huber_terms = [cp.huber(cp.norm(v[t, :]), M) for t in range(N)]
        obj = cp.Minimize(process_term + tau * cp.sum(huber_terms))

        # Constraints
        constraints = [x[0] == x0]
        for t in range(N):
            constraints.append(x[t + 1] == A @ x[t] + B @ w[t])
            constraints.append(y[t] == C @ x[t] + v[t])

        # Solve
        prob = cp.Problem(obj, constraints)
        # OSQP gives fast solutions for large QPs; fallback to ECOS otherwise
        try:
            prob.solve(solver=cp.OSQP, eps_abs=1e-6, eps_rel=1e-6, max_iter=20000)
        except Exception:
            try:
                prob.solve(solver=cp.ECOS, abstol=1e-6, reltol=1e-6)
            except Exception:
                return {"x_hat": [], "w_hat": [], "v_hat": []}

        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        return {
            "x_hat": x.value.tolist(),
            "w_hat": w.value.tolist(),
            "v_hat": v.value.tolist(),
        }
