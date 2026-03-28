from typing import Any
import numpy as np
import cvxpy as cp

class Solver:

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the robust Kalman filtering problem using the Huber loss function.
        """
        # Extract data
        A = np.array(problem['A'], dtype=float)
        B = np.array(problem['B'], dtype=float)
        C = np.array(problem['C'], dtype=float)
        y = np.array(problem['y'], dtype=float)
        x0 = np.array(problem['x_initial'], dtype=float)
        tau = float(problem['tau'])
        M = float(problem['M'])

        N, m = y.shape               # number of measurements
        n = A.shape[1]               # state dimension
        p = B.shape[1]               # process noise dimension

        # Decision variables
        x = cp.Variable((N + 1, n), name="x")
        w = cp.Variable((N, p), name="w")
        v = cp.Variable((N, m), name="v")

        # Objective: L2 process noise + Huber on measurement noise
        obj = cp.Minimize(cp.sum(cp.square(w)) + 
                          tau * cp.sum(cp.huber(cp.norm(v, axis=1), M)))

        # Constraints
        constraints = [x[0] == x0]
        constraints += [x[t + 1] == A @ x[t] + B @ w[t] for t in range(N)]
        constraints += [y[t] == C @ x[t] + v[t] for t in range(N)]

        # Solve
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.OSQP, eps_abs=1e-6, eps_rel=1e-6, max_iter=5000)

        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x.value is None:
            return {'x_hat': [], 'w_hat': [], 'v_hat': []}

        return {
            'x_hat': x.value.tolist(),
            'w_hat': w.value.tolist(),
            'v_hat': v.value.tolist()
        }