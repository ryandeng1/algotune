"""
solver.py
This file implements a fast robust Kalman filter solver based on CVXPY.
The solver uses the OSQP solver (or ECOS as fallback) for speed and
avoids unnecessary Python loops where possible.
"""

from typing import Any, Dict, List
import numpy as np
import cvxpy as cp


class Solver:
    """
    Robust Kalman filter solver using Huber loss for measurement noise.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve the robust Kalman filtering problem.

        Parameters
        ----------
        problem : dict
            Dictionary containing all problem data:
            - "A": state transition matrix (n x n)
            - "B": process noise matrix (n x p)
            - "C": measurement matrix (m x n)
            - "y": measurements array (N x m)
            - "x_initial": initial state vector (n,)
            - "tau": weight for measurement noise
            - "M": Huber threshold

        Returns
        -------
        dict
            Dictionary with keys "x_hat", "w_hat", "v_hat".
        """
        # Load data
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

        # Decision variables
        x = cp.Variable((N + 1, n), name="x")
        w = cp.Variable((N, p), name="w")
        v = cp.Variable((N, m), name="v")

        # Objective components
        process_term = cp.sum_squares(w)  # ||w_t||^2 summed over t
        # Huber term for each measurement v_t
        huber_terms = [tau * cp.huber(cp.norm(v[t, :]), M) for t in range(N)]
        measurement_term = cp.sum(huber_terms)

        objective = cp.Minimize(process_term + measurement_term)

        # Constraints
        constraints = [x[0] == x0]
        for t in range(N):
            constraints.append(x[t + 1] == A @ x[t] + B @ w[t])
            constraints.append(y[t] == C @ x[t] + v[t])

        # Problem
        prob = cp.Problem(objective, constraints)

        # Attempt to solve with OSQP first (fast for quadratic problems)
        try:
            prob.solve(solver=cp.OSQP, verbose=False, warm_start=True)
        except Exception:
            # Fallback to ECOS (more robust)
            try:
                prob.solve(solver=cp.ECOS, verbose=False, warm_start=True)
            except Exception:
                return {"x_hat": [], "w_hat": [], "v_hat": []}

        # Check solution validity
        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x.value is None:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        # Convert to python lists
        return {
            "x_hat": x.value.tolist(),
            "w_hat": w.value.tolist(),
            "v_hat": v.value.tolist(),
        }
