# solver.py
# Optimised solver implementation

from typing import Dict, Any
import numpy as np
import cvxpy as cp


class Solver:
    """
    Optimised solver for the time‑series optimization problem.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solves the following optimisation problem:

            min   ||w||^2  +  τ * ||v||^2
            s.t.  x[0]   = x_initial
                  x[t+1] = A @ x[t] + B @ w[t]      for t = 0..N-1
                  y[t]   = C @ x[t] + v[t]          for t = 0..N-1

        Parameters
        ----------
        problem : dict
            Dictionary containing the following keys:
                'A' : (n, n) array
                'B' : (n, p) array
                'C' : (m, n) array
                'y' : (N, m) array of observations
                'x_initial' : (n,) array
                'tau' : scalar penalty weight

        Returns
        -------
        dict
            Dictionary with keys 'x_hat', 'w_hat', 'v_hat' containing the
            optimal trajectories.  Empty lists are returned if the problem
            could not be solved.
        """

        # Convert input data to numpy arrays
        A = np.asarray(problem["A"])
        B = np.asarray(problem["B"])
        C = np.asarray(problem["C"])
        y = np.asarray(problem["y"])
        x0 = np.asarray(problem["x_initial"])
        tau = float(problem["tau"])

        N, m = y.shape
        n = A.shape[0]
        p = B.shape[1]

        # CVXPY variables
        x = cp.Variable((N + 1, n), name="x")
        w = cp.Variable((N, p), name="w")
        v = cp.Variable((N, m), name="v")

        # Objective
        objective = cp.Minimize(cp.norm_sqr(w) + tau * cp.norm_sqr(v))

        # Constraints
        constraints = [x[0] == x0]
        constraints += [
            x[t + 1] == A @ x[t] + B @ w[t] for t in range(N)
        ]
        constraints += [
            y[t] == C @ x[t] + v[t] for t in range(N)
        ]

        # Solve the problem with a fast solver
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.OSQP, warm_start=True, max_iters=2000)
        except (cp.SolverError, Exception):
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        # Validate solution
        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x.value is None:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        # Return result as nested Python lists
        return {
            "x_hat": x.value.tolist(),
            "w_hat": w.value.tolist(),
            "v_hat": v.value.tolist(),
        }