from __future__ import annotations
from typing import Any, Dict, List, Optional
import cvxpy as cp
import numpy as np


class Solver:
    def solve(
        self,
        problem: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Solve a soft‑margin SVM with CVXPY.

        The input dictionary must contain the keys:
          * `"X"` : a (n_samples × n_features) raw or pre‑processed feature matrix
          * `"y"` : a (n_samples,) list/array of labels in {−1, +1}
          * `"C"` : a positive penalty parameter

        The function returns a dictionary with the optimal parameters
        and simple metrics, or ``None`` if the problem could not be solved.
        """
        # Convert input data to NumPy arrays
        X = np.asarray(problem["X"], dtype=float)
        y = np.asarray(problem["y"], dtype=float).reshape(-1, 1)
        C = float(problem["C"])

        n_samples, n_features = X.shape

        # CVXPY decision variables
        beta = cp.Variable((n_features, 1))
        beta0 = cp.Variable()
        xi = cp.Variable((n_samples, 1))

        # Objective: 1/2 * ||beta||^2 + C * sum(xi)
        objective = cp.Minimize(0.5 * cp.sum_squares(beta) + C * cp.sum(xi))

        # Constraints:
        #   xi >= 0
        #   y_i (x_i^T beta + beta0) >= 1 - xi_i
        constraints = [
            xi >= 0,
            cp.multiply(y, X @ beta + beta0) >= 1 - xi,
        ]

        prob = cp.Problem(objective, constraints)

        try:
            optimal_value = prob.solve()
        except (cp.SolverError, Exception):
            return None

        # Check solver success
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return None

        if beta.value is None or beta0.value is None:
            return None

        # Compute predictions and misclassification rate
        pred = X @ beta.value + beta0.value
        missclass_error = float(np.mean((pred * y) < 0))

        return {
            "beta0": float(beta0.value),
            "beta": beta.value.squeeze().tolist(),
            "optimal_value": float(optimal_value),
            "missclass_error": missclass_error,
        }