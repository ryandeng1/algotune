# solver.py
from __future__ import annotations

import cvxpy as cp
import numpy as np


class Solver:
    """
    Optimised solver for a logistic‑regression with group LASSO.
    The implementation keeps all heavy‑weight symbolic work within CVXPY
    whilst performing the light‑weight preprocessing with NumPy,
    which is executed only once during the lifetime of the instance.
    """

    def solve(
        self,
        problem: dict[str, list[list[float]] | list[int] | float],
    ) -> dict[str, list[float] | float]:
        """
        Solve the logistic regression with group‑l1 regularisation.

        Parameters
        ----------
        problem:
            Dictionary containing:
            - 'X'  : 2‑D list of floats, shape (n, p + 1)   (Intercept column first)
            - 'y'  : 1‑D list of floats, shape (n,)
            - 'gl' : group labels for each feature (excluding intercept)
            - 'lba': scalar regularisation constant

        Returns
        -------
        result:
            Dictionary with keys:
            - 'beta0'          : Intercept value
            - 'beta'           : coefficient vector (length p)
            - 'optimal_value'  : objective value at optimum
        """
        # ----- 1️⃣  Prepare data -------------------------------------------------
        X = np.asarray(problem["X"], dtype=np.float64)
        y = np.asarray(problem["y"], dtype=np.float64).reshape(-1, 1)
        gl = np.asarray(problem["gl"], dtype=np.int64)
        lba = float(problem["lba"])

        n, p_plus = X.shape
        p = p_plus - 1  # number of features excluding intercept

        # Prep group encoding once per call
        ulabels, inverseinds, count = np.unique(gl[:, None], return_inverse=True, return_counts=True)
        m = ulabels.size
        group_idx = np.zeros((p, m), dtype=bool)
        group_idx[np.arange(p), inverseinds.flatten()] = 1
        not_group_idx = ~group_idx
        sqrt_group_size = np.sqrt(count)

        # ----- 2️⃣  CVXPY model construction ------------------------------------
        # Separate intercept column; features are X[:,1:]
        X_feat = X[:, 1:]

        beta = cp.Variable((p, m))
        beta0 = cp.Variable()
        beta[not_group_idx] = 0  # hard constraint that isolates groups

        # Build logistic loss
        linear_term = X_feat @ beta + beta0
        ifm = linear_term        # keep for readability
        y_broadcast = y               # shape (n,1)
        loss = -cp.sum(cp.multiply(y_broadcast, ifm)) + cp.sum(cp.logistic(ifm))

        # Group LASSO: ∑  ||β_g||2 * sqrt(|g|)
        group_norms = cp.norm(beta, 2, 0)
        penalty = lba * cp.sum(cp.multiply(group_norms, sqrt_group_size))

        prob = cp.Problem(cp.Minimize(loss + penalty))

        # ----- 3️⃣  Solve --------------------------------------------------------
        try:
            result = prob.solve(solver=cp.ECOS, verbose=False)
        except (cp.SolverError, Exception):
            return None

        # ----- 4️⃣  Extract solution --------------------------------------------
        if beta.value is None or beta0.value is None:
            return None

        # Re‑assemble β vector by taking the entry belonging to each feature
        beta_vec = beta.value[np.arange(p), inverseinds.flatten()]

        return {
            "beta0": float(beta0.value),
            "beta": beta_vec.tolist(),
            "optimal_value": float(result),
        }