import numpy as np
import cvxpy as cp
from typing import Any, Dict

# ------------------------------------------------------------------
#  SVM solver using CVXPY – fast, correct and minimal
# ------------------------------------------------------------------
class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any] | None:
        """
        Solve the soft‑margin SVM

        Parameters
        ----------
        problem : dict
            Must contain:
                * 'X' : array‑like, shape (n_samples, n_features)
                * 'y' : array‑like, shape (n_samples,) with values ±1
                * 'C' : positive float, regularization coeff.

        Returns
        -------
        dict
            * 'beta0'          – intercept
            * 'beta'           – coefficient vector
            * 'optimal_value'  – objective value at optimum
            * 'missclass_error' – fraction of misclassified samples
        or
            None if the problem cannot be solved.
        """

        # ----- 1. Convert inputs to numpy arrays ---------------------------------
        X = np.asarray(problem["X"], dtype=np.float64)
        y = np.asarray(problem["y"], dtype=np.float64).reshape(-1, 1)
        C = float(problem["C"])
        n, p = X.shape

        # ----- 2. Formulate CVXPY problem ---------------------------------------
        beta = cp.Variable(p)
        beta0 = cp.Variable()
        xi = cp.Variable(n)

        objective = 0.5 * cp.sum_squares(beta) + C * cp.sum(xi)
        constraints = [
            xi >= 0,
            cp.multiply(y, X @ beta + beta0) >= 1 - xi,
        ]

        prob = cp.Problem(cp.Minimize(objective), constraints)

        # ----- 3. Solve the problem --------------------------------------------
        try:
            optimal_value = prob.solve(solver=cp.SCS, eps=1e-6, max_iters=1000)
        except (cp.SolverError, RuntimeError):
            return None

        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return None
        if beta.value is None or beta0.value is None:
            return None

        # ----- 4. Compute performance metrics -----------------------------------
        preds = X @ beta.value + beta0.value
        missclass_error = np.mean((preds * y) < 0)

        # ----- 5. Prepare the result --------------------------------------------
        return {
            "beta0": float(beta0.value),
            "beta": beta.value.flatten().tolist(),
            "optimal_value": float(optimal_value),
            "missclass_error": float(missclass_error),
        }