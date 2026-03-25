import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[list[float]]:
        X = np.array(problem["X"])
        n_components = problem["n_components"]
        m, n = X.shape

        # Center the data
        X_centered = X - np.mean(X, axis=0)

        # Compute SVD with full_matrices=False for efficiency
        U, S, Vh = np.linalg.svd(X_centered, full_matrices=False)
        k = min(m, n)

        if n_components > k:
            V = np.eye(n_components, n)
        else:
            V = Vh.T
            V = V[:, :n_components].T

        return V.tolist()
