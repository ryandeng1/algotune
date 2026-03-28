from typing import Any
import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Convert inputs to NumPy arrays
        X = np.asarray(problem["X"], dtype=float)
        y = np.asarray(problem["y"], dtype=float).reshape(-1, 1)
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

        # Solve the problem
        prob = cp.Problem(objective, constraints)
        try:
            optimal_value = prob.solve(solver=cp.SCS, verbose=False)
        except Exception:
            return None

        if beta.value is None or beta0.value is None:
            return None

        # Compute predictions and missclass error
        preds = X @ beta.value + beta0.value
        missclass_error = float(np.mean(preds * y < 0))

        return {
            "beta0": float(beta0.value),
            "beta": beta.value.flatten().tolist(),
            "optimal_value": float(optimal_value),
            "missclass_error": missclass_error,
        }