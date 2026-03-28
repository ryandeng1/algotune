from typing import Any
import cvxpy as cp
import numpy as np


class Solver:
    def solve(self, problem: dict) -> dict:
        A = np.array(problem["covariance"])
        n_components = int(problem["n_components"])
        sparsity_param = float(problem["sparsity_param"])

        n = A.shape[0]
        X = cp.Variable((n, n_components))

        eigvals, eigvecs = np.linalg.eigh(A)
        pos_indices = eigvals > 0
        eigvals = eigvals[pos_indices]
        eigvecs = eigvecs[:, pos_indices]

        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        k = min(len(eigvals), n_components)
        B = eigvecs[:, :k] * np.sqrt(eigvals[:k])

        objective = cp.Minimize(cp.sum_squares(B - X) + sparsity_param * cp.norm1(X))
        constraints = [cp.norm(X[:, i]) <= 1 for i in range(n_components)]

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve()
            if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or X.value is None:
                return {"components": [], "explained_variance": []}
            
            components = X.value
            variances = np.diag(components.T @ A @ components)
            explained_variance = [float(v) for v in variances]
            return {"components": components.tolist(), "explained_variance": explained_variance}
        except cp.SolverError:
            return {"components": [], "explained_variance": []}
        except Exception:
            return {"components": [], "explained_variance": []}