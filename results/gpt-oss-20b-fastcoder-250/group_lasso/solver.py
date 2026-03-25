import numpy as np
import cvxpy as cp

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solves logistic regression with group lasso regularisation
        using CVXPY.  The implementation follows the reference
        solution, but is written to avoid unnecessary copies
        and intermediate variables for speed.
        """
        # Input extraction
        X = np.array(problem["X"], dtype=np.float64)
        y = np.array(problem["y"], dtype=np.float64)
        gl = np.array(problem["gl"], dtype=int)
        lba = float(problem["lba"])

        # Number of features and groups
        p = X.shape[1] - 1
        # unique groups and mapping
        ulabels, inverseinds, group_sizes = np.unique(gl, return_inverse=True, return_counts=True)
        m = ulabels.shape[0]
        # indicator matrix for group membership (p x m)
        group_matrix = np.zeros((p, m), dtype=int)
        group_matrix[np.arange(p), inverseinds] = 1
        # mask for non-group entries (to set to zero)
        not_group_mask = ~group_matrix.astype(bool)

        # Precompute sqrt of group sizes for weights
        sqrt_group_sizes = np.sqrt(group_sizes)

        # CVXPY variables
        beta = cp.Variable((p, m))
        beta0 = cp.Variable()
        # y reshaped for vectorized operations
        y_col = y.reshape(-1, 1)

        # Logistic loss
        Xb = X[:, 1:] @ beta  # n x m
        # sum across groups per sample
        linear = cp.sum(Xb, axis=1) + beta0
        logreg = -cp.sum(cp.multiply(y_col, linear)) + cp.sum(cp.logistic(linear))

        # Group lasso penalty (sum over groups of weight * ||beta_j||_2^2)
        squared_norms = cp.sum_squares(beta, axis=0)
        grouplasso = lba * cp.sum(cp.multiply(squared_norms, sqrt_group_sizes))
        objective = cp.Minimize(logreg + grouplasso)

        # Constraints: enforce zero on non-group entries
        constraints = [beta[not_group_mask] == 0]

        # Solve
        prob = cp.Problem(objective, constraints)
        try:
            result = prob.solve(solver=cp.ECOS, warm_start=True)
        except Exception:
            return None

        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return None
        if beta.value is None or beta0.value is None:
            return None

        # Extract beta as vector of length p
        beta_vec = beta.value[np.arange(p), inverseinds].ravel()

        return {
            "beta0": float(beta0.value),
            "beta": beta_vec.tolist(),
            "optimal_value": float(result)
        }
