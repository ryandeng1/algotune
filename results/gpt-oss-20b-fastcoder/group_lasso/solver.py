from typing import Any
import cvxpy as cp
import numpy as np

class Solver:
    """
    Optimised solver for logistic regression group lasso.
    """
    def solve(self, problem: dict[str, list[list[float]] | list[int] | float]) -> dict[str, list[float] | float] | None:
        # Convert input to numpy arrays only once
        X = np.asarray(problem['X'], dtype=np.float64)
        y = np.asarray(problem['y'], dtype=np.float64).reshape(-1, 1)
        gl = np.asarray(problem['gl'], dtype=np.int64)
        lba = float(problem['lba'])

        # Pre‑allocate useful values
        m = gl.size
        n_samples, n_features = X.shape
        p = n_features - 1                     # exclude intercept column
        pjs = np.bincount(gl, minlength=m).astype(np.float64)
        sqr_group_sizes = np.sqrt(pjs)

        # Build group mask matrix once (logical)
        inv = np.searchsorted(np.arange(m), gl)  # equivalent to return_inverse
        mask = np.zeros((p, m), dtype=bool)
        mask[np.arange(p), inv] = True
        not_group_mask = ~mask

        # Variables
        beta = cp.Variable((p, m))
        beta0 = cp.Variable()
        # Parameter for lambda (non‑negative)
        try:
            lba_cp = cp.Parameter(nonneg=True)
            lba_cp.value = lba
        except Exception:
            return None

        # Logistic regression part
        linear_term = X[:, 1:] @ beta
        logits = linear_term + beta0
        logreg = (
            -cp.sum(cp.multiply(y, logits)) + cp.sum(cp.logistic(logits))
        )

        # Group lasso penalty
        grouplasso = lba_cp * cp.sum(cp.multiply(cp.norm(beta, 2, 0), sqr_group_sizes))

        # Objective and constraints
        objective = cp.Minimize(logreg + grouplasso)
        constraints = [beta[not_group_mask] == 0]

        prob = cp.Problem(objective, constraints)

        # Solve once (default solver)
        try:
            result = prob.solve(solver=cp.ECOS, warm_start=True)
        except (cp.SolverError, Exception):
            return None

        # Extract results
        if beta.value is None or beta0.value is None:
            return None

        beta_extracted = beta.value[np.arange(p), inv]
        return {
            'beta0': float(beta0.value),
            'beta': beta_extracted.tolist(),
            'optimal_value': float(result)
        }