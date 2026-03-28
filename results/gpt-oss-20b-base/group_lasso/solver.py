from typing import Any
import numpy as np
import cvxpy as cp

class Solver:
    """
    Optimised logistic regression with group‑lasso.
    """
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        X = np.asarray(problem["X"], dtype=np.float64)
        y = np.asarray(problem["y"], dtype=np.float64).reshape(-1, 1)
        gl = np.asarray(problem["gl"], dtype=np.int64)  # shape (p,)
        lba = float(problem["lba"])

        p = X.shape[1] - 1  # number of regressors (excluding intercept)
        # Build a mapping from group label to the indices of its members
        group_indices = {}
        for idx, g in enumerate(gl):
            group_indices.setdefault(g, []).append(idx)

        # Variables
        beta = cp.Variable(p)
        beta0 = cp.Variable()

        # Logistic loss
        linear_part = X[:, 1:] @ beta + beta0          # shape (m,)
        logreg = -cp.sum(cp.multiply(y, linear_part)) + cp.sum(cp.logistic(linear_part))

        # Group lasso penalty
        group_penalty = 0
        for g, inds in group_indices.items():
            group_penalty += cp.norm(beta[inds], 2) * np.sqrt(len(inds))
        grouplasso = lba * group_penalty

        # Problem
        prob = cp.Problem(cp.Minimize(logreg + grouplasso))
        try:
            result = prob.solve(solver=cp.ECOS, max_iter=1000)
        except Exception:
            return None

        if beta.value is None or beta0.value is None:
            return None

        beta_val = beta.value.round()  # optional rounding
        return {
            "beta0": beta0.value,
            "beta": beta_val.tolist(),
            "optimal_value": result,
        }