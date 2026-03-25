# solver.py
import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solve the logistic regression with a group Lasso penalty.

        Parameters
        ----------
        problem : dict
            Should contain the keys:
                - "X" : 2D array of shape (n, p+1) where the first column
                         is the intercept column of all ones.
                - "y" : 1D array of labels containing only 0 or 1.
                - "gl": 1D array of length p where gl[i] specifies the
                         group (integer) of feature i+1.
                - "lba": float, the regularisation parameter λ.

        Returns
        -------
        dict
            A dictionary with keys
                - "beta0"        : intercept scalar
                - "beta"         : list of length p containing the
                                   estimated coefficients (excluding
                                   the intercept)
                - "optimal_value": the objective value at the optimum.
        """
        # Convert inputs to numpy arrays
        X = np.asarray(problem["X"], dtype=np.float64)
        y = np.asarray(problem["y"], dtype=np.float64)
        gl = np.asarray(problem["gl"], dtype=int)
        lba = float(problem["lba"])

        # Preprocess group information
        # Get unique group labels and mapping from features to group
        unique_groups, inverse, group_sizes = np.unique(gl, return_inverse=True, return_counts=True)
        p = X.shape[1] - 1          # number of non‑intercept features
        n = X.shape[0]              # number of samples
        G = unique_groups.size     # number of groups

        # Build a binary design matrix for group membership (p × G)
        group_matrix = np.zeros((p, G), dtype=int)
        group_matrix[np.arange(p), inverse] = 1
        not_group = ~group_matrix.astype(bool)

        # The weight for each group in the regulariser
        w = np.sqrt(group_sizes).astype(np.float64)

        # CVXPY variables
        beta = cp.Variable((p, G))
        beta0 = cp.Variable()
        # Log‑likelihood part of the objective
        xb = X[:, 1:] @ beta   # shape (n, G)
        linear = xb @ np.ones(G) + beta0
        logreg = -cp.sum(cp.multiply(y, linear)) + cp.sum(cp.logistic(linear))
        # Group lasso penalty
        penalty = lba * cp.sum(cp.multiply(cp.norm(beta, 2, 0), w))
        objective = cp.Minimize(logreg + penalty)

        # Enforce that each feature belongs only to its group
        constraints = [beta[not_group] == 0]

        # Solve the problem
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.ECOS, verbose=False, abstol=1e-5, reltol=1e-5)
        except Exception:
            return None

        # Check solver status
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return None

        # Extract the estimated coefficients
        beta_val = beta.value[np.arange(p), inverse]
        return {
            "beta0": beta0.value,
            "beta": beta_val.tolist(),
            "optimal_value": prob.value
        }
