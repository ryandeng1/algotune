from typing import Any, Dict, List, Union
import numpy as np

import cvxpy as cp

class Solver:
    def solve(
        self,
        problem: Dict[str, Union[List[List[float]], List[int], float]]
    ) -> Dict[str, Union[List[float], float, None]]:
        """
        Solves a group‑lasso regularised logistic regression using CVXPY.

        Parameters
        ----------
        problem : dict
            Dictionary containing:
              * "X"   : 2‑D list of shape (n_samples, n_features+1) – the first column
                        corresponds to the bias term and is therefore ignored.
              * "y"   : list of labels (±1) of length n_samples.
              * "gl"  : list of group labels for each feature (length n_features).
              * "lba" : non‑negative scalar, regularisation weight λ.

        Returns
        -------
        dict
            * "beta0"      – bias term.
            * "beta"       – list of regression coefficients, one per feature.
            * "optimal_value" – value of the objective at the optimum.
        """
        # ---------------------------------------------------------------------
        # 1. Convert arguments to NumPy objects
        # ---------------------------------------------------------------------
        X = np.asarray(problem["X"], dtype=np.float64)   # (n, p+1)
        y = np.asarray(problem["y"], dtype=np.float64)   # (n,)
        gl = np.asarray(problem["gl"], dtype=np.int64)   # (p,)
        lba = float(problem["lba"])

        n, p_plus = X.shape
        p = p_plus - 1  # number of features (bias excluded)

        # ---------------------------------------------------------------------
        # 2. Prepare group encoding
        # ---------------------------------------------------------------------
        groups, inv_idx, group_counts = np.unique(gl, return_inverse=True, return_counts=True)
        m = groups.size                         # number of distinct groups

        # Each feature belongs to exactly one group.
        # Create a binary mask (p, m) where mask[i, j] = 1 if feature i is in group j
        group_mask = np.zeros((p, m), dtype=bool)
        group_mask[np.arange(p), inv_idx] = True
        nongroup_mask = ~group_mask

        group_norm_weights = np.sqrt(group_counts).astype(np.float64)

        # ---------------------------------------------------------------------
        # 3. CVXPY variables & problem definition
        # ---------------------------------------------------------------------
        beta = cp.Variable((p, m))
        beta0 = cp.Variable()
        Wbeta = cp.multiply(cp.norm(beta, 2, 0), group_norm_weights)  # group weights
        logreg = (        # logistic loss
            -cp.sum(cp.multiply(y, X[:, 1:] @ beta + beta0))
            + cp.sum(cp.logistic(X[:, 1:] @ beta + beta0))
        )
        obj = cp.Minimize(logreg + lba * cp.sum(Wbeta))
        constraints = [beta[nongroup_mask] == 0]

        prob = cp.Problem(obj, constraints)

        # ---------------------------------------------------------------------
        # 4. Solve
        # ---------------------------------------------------------------------
        result = None
        try:
            result = prob.solve(verbose=False, warm_start=True)
        except (cp.SolverError, Exception):
            return None

        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            return None

        if beta.value is None or beta0.value is None:
            return None

        # ---------------------------------------------------------------------
        # 5. Map solution back to feature space
        # ---------------------------------------------------------------------
        beta_vals = beta.value[np.arange(p), inv_idx]  # shape (p,)

        return {
            "beta0": float(beta0.value),
            "beta": beta_vals.tolist(),
            "optimal_value": float(result),
        }