from typing import Any
import cvxpy as cp
import numpy as np


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Solve a robust Kalman filtering problem with Huber loss."""
        # Convert inputs to NumPy arrays
        A = np.array(problem["A"], dtype=float)
        B = np.array(problem["B"], dtype=float)
        C = np.array(problem["C"], dtype=float)
        y = np.array(problem["y"], dtype=float)
        x0 = np.array(problem["x_initial"], dtype=float)
        tau = float(problem["tau"])
        M = float(problem["M"])

        N, m = y.shape
        n = A.shape[1]
        p = B.shape[1]

        # Decision variables
        x = cp.Variable((N + 1, n), name="x")
        w = cp.Variable((N, p), name="w")
        v = cp.Variable((N, m), name="v")

        # Objective: squared process noise + weighted Huber loss on measurement residuals
        process_noise = cp.sum_squares(w)
        # Vectorise Huber computation over all time steps
        huber_terms = cp.huber(cp.norm(v, axis=1), M)
        measurement_noise = tau * cp.sum(huber_terms)

        objective = cp.Minimize(process_noise + measurement_noise)

        # Constraints
        constraints = [x[0] == x0]
        constraints += [
            x[t + 1] == A @ x[t] + B @ w[t]
            for t in range(N)
        ]
        constraints += [
            y[t] == C @ x[t] + v[t]
            for t in range(N)
        ]

        # Solve the problem
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
        except Exception:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        # Handle infeasible or unbounded cases
        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x.value is None:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        # Return results as Python lists
        return {
            "x_hat": x.value.tolist(),
            "w_hat": w.value.tolist(),
            "v_hat": v.value.tolist(),
        }