from typing import Any
import cvxpy as cp
import numpy as np


class Solver:
    """
    A lightweight wrapper around CVXPY for solving a group‑lasso logistic regression.
    Most of the time is spent in the underlying convex solver, so the main goal is
    to avoid unnecessary Python & NumPy overhead.
    """

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the logistic regression with group‑lasso penalty.

        Parameters
        ----------
        problem : dict
            Must contain the keys:
                - 'X'  : (N, D) list of lists (or ndarray)
                - 'y'  : (N,) list of 0/1 indicator
                - 'gl' : (p,) list of group labels (p = D-1)
                - 'lba': group‑lasso regulariser constant

        Returns
        -------
        dict
            Dictionary with keys:
                'beta0'        : intercept
                'beta'         : list of group coefficients (p,) (groups merged)
                'optimal_value': value of the objective at optimality
            or None if the problem could not be solved.
        """
        # 1. Prepare data -------------------------------------------------------
        X = np.asarray(problem["X"], dtype=np.float64, order="C")
        y = np.asarray(problem["y"], dtype=np.float64, order="C").reshape(-1, 1)
        gl = np.asarray(problem["gl"], dtype=np.int64, order="C")
        lba = float(problem["lba"])

        N, D = X.shape
        p = D - 1                      # number of coefficient columns
        unique, inv, cnt = np.unique(gl, return_inverse=True, return_counts=True)
        m = unique.size

        # Build a boolean matrix that indicates whether a coefficient belongs to a group
        # shape (p, m).  Using advanced indexing yields a tiny Python cost.
        group_idx = np.zeros((p, m), dtype=bool, order="C")  # all False initially
        group_idx[np.arange(p), inv] = True

        # Groups we keep (zeros are removed later, so keep mask is the inverse)
        not_group_idx = ~group_idx

        # Weight for each group: sqrt of size
        sqrt_group_sizes = np.sqrt(cnt, dtype=np.float64)

        # 2. Build CVXPY objects -----------------------------------------------
        # Variables
        beta = cp.Variable((p, m), name="beta")   # (p groups per coefficient)
        beta0 = cp.Variable(name="beta0")

        # Parameter (set once, no need to re‑initialise each solve)
        lba_param = cp.Parameter(nonneg=True, constant=True)
        lba_param.value = lba

        # Logistic regression part (vectorised)
        linear_term = X[:, 1:] @ beta          # (N, m)
        # Sum over groups inside each data point
        sum_over_groups = cp.sum(linear_term, axis=1, keepdims=True)  # (N, 1)

        logreg = (
            -cp.sum(cp.multiply(y, sum_over_groups + beta0))
            + cp.sum(cp.logistic(sum_over_groups + beta0))
        )

        # Group LASSO penalty
        # ``cp.norm(beta, 2, 0)`` gives the ℓ₂ norm for each column; summing over columns
        # and weighting by group size implements the group‑lasso.
        grouplasso = lba_param * cp.sum(cp.multiply(cp.norm(beta, 2, 0), sqrt_group_sizes))

        # Objective
        objective = cp.Minimize(logreg + grouplasso)

        # Constraints: force zero coefficients for missing groups
        constraints = [beta[not_group_idx] == 0]

        # 3. Solve -------------------------------------------------------------
        problem_obj = cp.Problem(objective, constraints)

        try:
            opt_val = problem_obj.solve(solver=cp.SCS, verbose=False, eps=1e-8)
        except Exception:
            return None

        # 4. Retrieve solution --------------------------------------------------
        if beta.value is None or beta0.value is None:
            return None

        # Collapse group matrix back to a single coefficient per variable
        beta_vector = beta.value[np.arange(p), inv]
        return {
            "beta0": float(beta0.value),
            "beta": beta_vector.tolist(),
            "optimal_value": float(opt_val),
        }