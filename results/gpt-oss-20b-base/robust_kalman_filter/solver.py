from typing import Any
import cvxpy as cp
import numpy as np


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the robust Kalman filtering problem using the Huber loss function.

        Parameters
        ----------
        problem : dict
            Dictionary with system matrices, measurements, and parameters.

        Returns
        -------
        dict
            Dictionary with estimated states and noise sequences.
        """
        # Parse input
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

        # Variables
        x = cp.Variable((N + 1, n), name="x")
        w = cp.Variable((N, p), name="w")
        v = cp.Variable((N, m), name="v")

        # Objective
        process_noise = cp.sum_squares(w)
        measurement_noise = tau * cp.sum(cp.huber(v, M))
        obj = cp.Minimize(process_noise + measurement_noise)

        # Constraints
        constraints = [x[0] == x0]
        for t in range(N):
            constraints.append(x[t + 1] == A @ x[t] + B @ w[t])
            constraints.append(y[t] == C @ x[t] + v[t])

        # Solve
        prob = cp.Problem(obj, constraints)
        try:
            prob.solve()
        except cp.SolverError:
            return {"x_hat": [], "w_hat": [], "v_hat": []}
        except Exception:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x.value is None:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        return {
            "x_hat": x.value.tolist(),
            "w_hat": w.value.tolist(),
            "v_hat": v.value.tolist(),
        }