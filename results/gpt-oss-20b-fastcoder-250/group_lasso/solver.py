from __future__ import annotations

import numpy as np
import cvxpy as cp
from typing import Dict, List, Union


class Solver:
    """
    Solves a logistic‑regression group‑lasso problem with CVXPY.
    The problem is defined as follows::

        min  g(β) + λ ∑_{j=1}^J w_j ||β_{(j)}||_2^2
        s.t.  β[:, j] == 0,  when feature i does not belong to group j

    Where
        g(β) = -∑_{i=1}^n y_i (Xβ)_i + ∑_{i=1}^n log(1 + exp((Xβ)_i)).

    Parameters
    ----------
    problem:
        Dict mapping the keys ``"X"``, ``"y"``, ``"gl"``, ``"lba"`` to
        numpy‑array compatible data.  ``X`` must be a 2‑D array whose
        first column is the intercept (i.e., a column of ones).  ``gl``
        maps each feature (excluding the intercept) to a group index.
    """

    def solve(
        self, problem: Dict[str, Union[List[List[float]], List[int], float]]
    ) -> Dict[str, Union[List[float], float]] | None:
        # Convert inputs to numpy arrays.  The casting is cheap because
        # the data are small relative to the CP variable size.
        X = np.asarray(problem["X"], dtype=float)
        y = np.asarray(problem["y"], dtype=float).reshape(-1, 1)
        gl = np.asarray(problem["gl"], dtype=int)
        lba = float(problem["lba"])

        # Group information --------------------------------------------------
        # ``unique`` gives the unique group labels,  the inverse indices, and the
        # counts.  We only need the counts for the group penalty weights.
        _, inv_idx, group_counts = np.unique(gl, return_inverse=True, return_counts=True)

        n_features = X.shape[1] - 1         # exclude intercept
        n_groups = group_counts.shape[0]

        # Build a binary mask that indicates which feature belongs to which
        # group.  This is a sparse representation but CVXPY can handle it directly.
        group_mask = np.zeros((n_features, n_groups), dtype=bool)
        group_mask[np.arange(n_features), inv_idx] = True

        # Indices of features that belong *not* to a group – these must be
        # zeroed out in the CVXPY variables.
        non_group_idx = ~group_mask

        # Normalization factors for the group lasso penalty.
        sqrt_group_size = np.sqrt(group_counts)

        # --- CVXPY setup ----------------------------------------------------
        beta = cp.Variable((n_features, n_groups))
        beta0 = cp.Variable()
        logistic_term = (
            -cp.sum(cp.multiply(y, X[:, 1:] @ beta + beta0))
            + cp.sum(cp.logistic(X[:, 1:] @ beta + beta0))
        )
        group_lasso = lba * cp.sum(cp.multiply(cp.norm(beta, 2, 0), sqrt_group_size))
        objective = cp.Minimize(logistic_term + group_lasso)

        constraints = [beta[non_group_idx] == 0]
        prob = cp.Problem(objective, constraints)

        # Solve and handle common CVXPY errors gracefully
        try:
            opt_val = prob.solve()
        except (cp.SolverError, Exception):
            return None

        # Verify successful solution
        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            return None
        if beta.value is None or beta0.value is None:
            return None

        # Extract the coefficient estimates in dense format.
        beta_dense = beta.value[np.arange(n_features), inv_idx]

        return {
            "beta0": beta0.value,
            "beta": beta_dense.tolist(),
            "optimal_value": opt_val,
        }