import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve logistic regression with group lasso penalty.
        A lightweight implementation that uses iterative proximal gradient descent
        and avoids the heavy CVXPY backend. The algorithm is scalable for typical
        problem sizes encountered in the benchmark.
        """
        # Extract problem data
        X = np.asarray(problem["X"], dtype=np.float64)
        y = np.asarray(problem["y"], dtype=np.float64)
        gl = np.asarray(problem["gl"], dtype=np.int64)
        lba = float(problem["lba"])

        # Number of samples, features (including intercept column)
        n, d = X.shape
        # Features excluding intercept (first column is assumed to be 1)
        X_f = X[:, 1:]

        # Group information
        uniq_groups, inv_idx, group_sizes = np.unique(gl, return_inverse=True, return_counts=True)
        num_groups = uniq_groups.size
        p = d - 1

        # Build sparse indicator matrix for groups (one-hot encoding of groups over features)
        group_mask = np.zeros((p, num_groups), dtype=np.float64)
        group_mask[np.arange(p), inv_idx] = 1.0

        # Pre‑compute group norms with square‑root scaling
        sqrt_group_sizes = np.sqrt(group_sizes)

        # Initialise coefficients; start from zero
        beta = np.zeros((p, num_groups), dtype=np.float64)
        beta0 = 0.0

        # Step sizes
        lr = 0.01  # small learning rate
        num_iters = 2000

        # Helper views for efficient updates
        def group_norms(b):
            return np.linalg.norm(b, axis=0) * sqrt_group_sizes

        # Proximal operator for group lasso (soft group thresholding)
        def prox_group_lasso(b, thresh):
            norms = np.linalg.norm(b, axis=0)
            shrink = np.maximum(0, norms - thresh) / (norms + 1e-12)
            return b * shrink

        # Logistic regression gradient w.r.t intercept and beta
        def grad_intercept_beta(b0, B):
            linear = X_f @ (B @ group_mask.T) + b0
            prob = 1.0 / (1.0 + np.exp(-linear))
            grad_b0 = np.sum(prob - y)
            grad_B = (X_f.T @ ((prob - y)[:, None] * group_mask.T)).T
            return grad_b0, grad_B

        # Main optimization loop
        for _ in range(num_iters):
            grad_b0, grad_B = grad_intercept_beta(beta0, beta)
            beta0 -= lr * grad_b0
            beta -= lr * grad_B
            # Proximal step for each group
            beta = prox_group_lasso(beta, lr * lba)

        # Flatten beta to a vector according to original feature ordering
        beta_vec = np.empty(p, dtype=np.float64)
        beta_vec = beta[:, inv_idx]
        return {
            "beta0": float(beta0),
            "beta": beta_vec.tolist(),
            "optimal_value": None,  # not computed in this simple implementation
        }