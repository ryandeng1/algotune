from typing import Any
import numpy as np
import cvxpy as cp


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Robust Kalman filtering using a Huber loss on measurement noise.
        """
        # Convert problem data to numpy arrays
        A = np.asarray(problem["A"], dtype=float)
        B = np.asarray(problem["B"], dtype=float)
        C = np.asarray(problem["C"], dtype=float)
        y = np.asarray(problem["y"], dtype=float)
        x0 = np.asarray(problem["x_initial"], dtype=float)
        tau = float(problem["tau"])
        M = float(problem["M"])

        N, m = y.shape                      # number of time steps, measurement dimension
        n = A.shape[1]                      # state dimension
        p = B.shape[1]                      # process noise dimension

        # Variable declaration
        x = cp.Variable((N + 1, n))
        w = cp.Variable((N, p))
        v = cp.Variable((N, m))

        # Objective: sum of squared process noise + Huber on measurement noise
        process_term = cp.sum_squares(w)
        # Vectorised Huber: apply to each row norm of v
        measurement_term = tau * cp.sum(cp.huber(cp.norm(v, axis=1), M))
        objective = cp.Minimize(process_term + measurement_term)

        # Constraints
        constraints = [x[0] == x0]
        for t in range(N):
            constraints.append(x[t + 1] == A @ x[t] + B @ w[t])
            constraints.append(y[t] == C @ x[t] + v[t])

        # Solve
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.OSQP)         # use a fast default solver
        except cp.SolverError:
            return {"x_hat": [], "w_hat": [], "v_hat": []}
        except Exception:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x.value is None:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        return {
            "x_hat": x.value.tolist(),
            "w_hat": w.value.tolist(),
            "v_hat": v.value.tolist(),
        }