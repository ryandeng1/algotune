import numpy as np
import cvxpy as cp
from typing import Any, Dict, List, Union

# --------------------------------------------------------------- #
#        Efficient CVXPY code for Logistic Group‑Lasso          #
# --------------------------------------------------------------- #

class Solver:
    """
    Optimised solver for logistic regression with group L1 regularisation.
    """

    def solve(
        self,
        problem: Dict[str, Union[List[List[float]], List[int], float]]
    ) -> Union[Dict[str, Union[List[float], float]], None]:
        """
        Parameters
        ----------
        problem : dict
            Keys
                * X   : design matrix (p+1 x n)
                * y   : response vector (n,)
                * gl  : group labels (p+1,) integer array
                * lba : regularisation weight (float)

        Returns
        -------
        dict or None
            * beta0          : intercept
            * beta           : coefficient vector
            * optimal_value : objective value of the optimal solution
            None if solver fails
        """

        # -------- fast data prep --------------------------------
        X = np.asarray(problem["X"], dtype=np.float64, order="F")  # For memory‑layout
        y = np.asarray(problem["y"], dtype=np.float64, order="F")
        gl = np.asarray(problem["gl"], dtype=np.int64, order="F")
        lba: float = problem["lba"]

        # Sub‑sample index (exclude intercept if any)
        p = X.shape[0] - 1                     # number of covariates
        X_cov = X[1:, :]                       # Drop the first row (intercept column)
        y = y.reshape(-1, 1)                   # column vector

        # Unique groups and mapping from coefficients to groups
        uniq, inv = np.unique(gl[1:], return_inverse=True)  # ignore intercept group
        ng = uniq.size

        # ----- construct group masks for constraints -----------------
        # mask[j, g] = 1 if coefficient j belongs to group g else 0
        mask = np.zeros((p, ng), dtype=np.bool_)
        mask[np.arange(p), inv] = True
        db = np.logical_not(mask)

        #----- CVXPY variables --------------------------------------
        beta = cp.Variable((p, ng))       # beta[j,g]
        beta0 = cp.Variable()             # intercept
        # a workaround to allow setting the parameter externally
        lba_param = cp.Parameter(nonneg=True)
        lba_param.value = lba

        # ----- objective --------------------------------------------
        # logistic loss
        lin_pred = (X_cov.T @ beta)        # shape (n, ng)
        lin_pred_sum = cp.sum(lin_pred, axis=1, keepdims=True) + beta0
        logreg = -cp.sum(cp.multiply(y, lin_pred_sum)) + cp.sum(cp.logistic(lin_pred_sum))

        # group lasso penalty: sum over groups of l2 norms of vector of coeffs
        grp_norms = cp.norm(beta, 2, axis=0)   # shape (ng,)
        grouplasso = lba_param * cp.sum(grp_norms * np.sqrt(uniq.size))  # weight by sqrt(size)

        objective = cp.Minimize(logreg + grouplasso)

        # ----- constraints ------------------------------------------
        constraints = [beta[db] == 0]   # zero out coefficients not in their group

        # ----- solve --------------------------------------------------
        prob = cp.Problem(objective, constraints)
        try:
            opt_val = prob.solve(solver=cp.ECOS, warm_start=True)
        except Exception:
            return None

        # ----- return results ---------------------------------------
        if beta.value is None or beta0.value is None:
            return None

        # reconstruct coefficient vector in original order
        beta_vec = np.empty(p, dtype=np.float64)
        beta_vec[:] = beta.value[np.arange(p), inv]
        return {
            "beta0": beta0.value,
            "beta": beta_vec.tolist(),
            "optimal_value": opt_val,
        }