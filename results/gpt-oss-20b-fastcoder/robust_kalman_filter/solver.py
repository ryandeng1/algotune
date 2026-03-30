from __future__ import annotations

import cvxpy as cp
import numpy as np
from typing import Any, Dict


class Solver:
    """
    A lightweight implementation of a robust Kalman filtering problem.
    The formulation is a quadratic program with Huber loss on the measurement residual.
    """

    @staticmethod
    def _to_np(value: Any) -> np.ndarray:
        """Safe conversion to a NumPy array."""
        return np.asarray(value, dtype=float)

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve the robust Kalman filtering problem using the Huber loss function.
        Parameters
        ----------
        problem : dict
            Dictionary containing:
                A : array-like, shape (n, n)
                B : array-like, shape (n, p)
                C : array-like, shape (m, n)
                y : array-like, shape (N, m)
                x_initial : array-like, shape (n,)
                tau : float
                M : float
        Returns
        -------
        dict
            Dictionary containing solution arrays:
                x_hat : list[list[float]]
                w_hat : list[list[float]]
                v_hat : list[list[float]]
        """
        # ------------------------------------------------------------------
        # Extract problem data (avoids Python loops)
        # ------------------------------------------------------------------
        A = self._to_np(problem["A"])
        B = self._to_np(problem["B"])
        C = self._to_np(problem["C"])
        y = self._to_np(problem["y"])
        x0 = self._to_np(problem["x_initial"])
        tau = float(problem["tau"])
        M = float(problem["M"])

        N, m = y.shape           # number of time-steps and measurement dimension
        n = A.shape[1]           # state dimension
        p = B.shape[1]           # process noise dimension

        # ------------------------------------------------------------------
        # CVXPY variables
        # ------------------------------------------------------------------
        x = cp.Variable((N + 1, n), name="x")   # state trajectory
        w = cp.Variable((N, p), name="w")       # process noise
        v = cp.Variable((N, m), name="v")       # measurement noise

        # ------------------------------------------------------------------
        # Objective: quadratic process noise + Huber residual cost
        # ------------------------------------------------------------------
        process_noise_term = cp.sum_squares(w)
        # Vectorised Huber on the 1‑norm of each residual: ||v[t,:]||₂
        residual_norms = cp.norm(v, axis=1)
        measurement_noise_term = tau * cp.sum(cp.huber(residual_norms, M))
        obj = cp.Minimize(process_noise_term + measurement_noise_term)

        # ------------------------------------------------------------------
        # Constraints
        # ------------------------------------------------------------------
        constraints = [x[0] == x0]
        # System dynamics and measurement equations
        for t in range(N):
            constraints.append(x[t + 1] == A @ x[t] + B @ w[t])
            constraints.append(y[t] == C @ x[t] + v[t])

        # ------------------------------------------------------------------
        # Solve using the default solver (OSQP / ECOS for QP)
        # ------------------------------------------------------------------
        prob = cp.Problem(obj, constraints)
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
        except cp.SolverError:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        # ------------------------------------------------------------------
        # Return the solution
        # ------------------------------------------------------------------
        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x.value is None:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        return {
            "x_hat": x.value.tolist(),
            "w_hat": w.value.tolist(),
            "v_hat": v.value.tolist(),
        }