# solver.py
"""
Optimised SVM solver that uses CVXPY with the OSQP solver for maximum speed.
"""

from typing import Any, Dict

import cvxpy as cp
import numpy as np


class Solver:
    # We pre‑compile the OSQP solver options only once.
    _osqp_opts = {"scale": "normal", "verbose": False, "max_iter": 2000, "eps_abs": 1e-6, "eps_rel": 1e-6}

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve an L2‑regularised SVM via CVXPY.

        Parameters
        ----------
        problem : dict
            Dictionary containing keys
                - 'X': array‑like n×p training matrix
                - 'y': array‑like n‑label vector (±1)
                - 'C': float regularisation coefficient

        Returns
        -------
        dict
            Dictionary containing
                - 'beta0'        : scalar bias term
                - 'beta'         : list of p weight coefficients
                - 'optimal_value': value of the objective function
                - 'missclass_error': fraction of mis‑classified training samples
        """
        # Convert input to numpy arrays once
        X = np.asarray(problem["X"], dtype=np.float64)
        y = np.asarray(problem["y"], dtype=np.float64).reshape(-1, 1)
        C = float(problem["C"])

        n, p = X.shape

        # CVXPY variables
        beta = cp.Variable((p, 1))
        beta0 = cp.Variable()
        xi = cp.Variable((n, 1))

        # Objective and constraints
        objective = cp.Minimize(0.5 * cp.sum_squares(beta) + C * cp.sum(xi))
        constraints = [xi >= 0,
                       cp.multiply(y, X @ beta + beta0) >= 1 - xi]

        # Solve using OSQP for speed
        prob = cp.Problem(objective, constraints)
        try:
            optimal_value = prob.solve(solver=cp.OSQP, **self._osqp_opts)
        except Exception:
            # Any solving issue -> return None
            return None

        # Check feasibility
        if beta.value is None or beta0.value is None:
            return None

        # Predictions and miss‑classification rate
        predictions = X @ beta.value + beta0.value
        missclass = np.mean(predictions * y < 0)

        return {
            "beta0": float(beta0.value),
            "beta": beta.value.ravel().tolist(),
            "optimal_value": float(optimal_value),
            "missclass_error": float(missclass),
        }