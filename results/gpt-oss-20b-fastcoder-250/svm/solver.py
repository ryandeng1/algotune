# solver.py
import numpy as np
import cvxpy as cp
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        Solves the soft‑margin SVM optimisation problem

            min  1/2 ||beta||^2 + C sum xi
            s.t. xi >= 0
                 y_i (x_i^T beta + beta0) >= 1 - xi

        Parameters
        ----------
        problem : dict
            Dictionary containing the keys 'X', 'y' and 'C'.
            - X : n x p array of feature vectors
            - y : list of length n with entries -1 or 1
            - C : positive regularisation constant

        Returns
        -------
        dict
            Keys:
                'beta0'          : intercept
                'beta'           : coefficient vector (list)
                'optimal_value'  : objective value
                'missclass_error': misclassification rate
        """
        # Convert input to numpy arrays
        X = np.asarray(problem["X"], dtype=np.float64)
        y = np.asarray(problem["y"], dtype=np.float64).reshape(-1, 1)
        C = float(problem["C"])
        n, p = X.shape

        # Define variables
        beta = cp.Variable((p, 1), name="beta")
        beta0 = cp.Variable(name="beta0")
        xi = cp.Variable((n, 1), name="xi")

        # Objective: 0.5*||beta||^2 + C sum xi
        objective = cp.Minimize(0.5 * cp.sum_squares(beta) + C * cp.sum(xi))

        # Constraints
        constraints = [
            xi >= 0,
            cp.multiply(y, X @ beta + beta0) >= 1 - xi
        ]

        # Form and solve problem
        prob = cp.Problem(objective, constraints)
        # Use a fast solver that is available in most environments
        prob.solve(solver=cp.ECOS, warm_start=True, verbose=False)

        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return None

        beta_val = beta.value
        beta0_val = beta0.value

        if beta_val is None or beta0_val is None:
            return None

        # Compute predictions and misclassification error
        preds = (X @ beta_val + beta0_val).flatten()
        misclass = np.mean(np.sign(preds) != y.flatten())

        return {
            "beta0": float(beta0_val),
            "beta": beta_val.flatten().tolist(),
            "optimal_value": float(prob.value),
            "missclass_error": float(misclass)
        }
