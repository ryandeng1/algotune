from typing import Any
import cvxpy as cp
import numpy as np

class Solver:

    def solve(self, problem: dict[str, list[list[float]] | list[int] | float]) -> dict[str, list[float] | float]:
        """
        Solves the logistic regression group lasso using CVXPY.
        """
        # Convert inputs to numpy arrays
        X = np.array(problem['X'])
        y = np.array(problem['y'])
        gl = np.array(problem['gl'])
        lba = problem['lba']

        # Decode group labels
        ulabels, inverseinds, pjs = np.unique(gl[:, None], return_inverse=True, return_counts=True)

        # Dimensions
        p = X.shape[1] - 1  # number of features (excluding bias)
        m = ulabels.shape[0]  # number of unique groups

        # Build group index mask: rows correspond to features, columns to groups
        group_idx = np.zeros((p, m))
        group_idx[np.arange(p), inverseinds.flatten()] = 1
        not_group_idx = ~group_idx
        sqr_group_sizes = np.sqrt(pjs)

        # CVXPY variables and parameter
        beta = cp.Variable((p, m))
        beta0 = cp.Variable()
        lbacp = cp.Parameter(nonneg=True)

        # Logistic regression objective
        y_col = y[:, None]
        linear_term = X[:, 1:] @ beta
        logreg = (
            -cp.sum(cp.multiply(y_col, cp.sum(linear_term, 1, keepdims=True) + beta0))
            + cp.sum(cp.logistic(cp.sum(linear_term, 1) + beta0))
        )

        # Group lasso penalty
        grouplasso = lba * cp.sum(cp.multiply(cp.norm(beta, 2, 0), sqr_group_sizes))
        objective = cp.Minimize(logreg + grouplasso)

        # Enforce zero entries outside groups
        constraints = [beta[not_group_idx] == 0]
        lbacp.value = lba

        # Solve the problem
        prob = cp.Problem(objective, constraints)
        try:
            result = prob.solve(solver=cp.ECOS, warm_start=True, verbose=False)
        except (cp.SolverError, Exception):
            return None

        # Check if solution is valid
        if beta.value is None or beta0.value is None:
            return None

        # Reshape solution to match expected format
        beta_reshaped = beta.value[np.arange(p), inverseinds.flatten()]
        return {
            'beta0': float(beta0.value),
            'beta': beta_reshaped.tolist(),
            'optimal_value': float(result)
        }