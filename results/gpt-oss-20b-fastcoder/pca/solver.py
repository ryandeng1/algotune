import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[list[float]]:
        X = np.asarray(problem['X'], dtype=float)
        n_components = problem['n_components']
        # Center the data
        X -= X.mean(axis=0)
        # Compute SVD
        try:
            U, S, Vt = np.linalg.svd(X, full_matrices=False, compute_uv=True)
        except Exception:
            # Fallback to identity if SVD fails
            d = X.shape[1]
            V = np.eye(d, d)[:n_components]
            return V.tolist()
        # Return the first n_components rows of Vt
        V = Vt[:n_components]
        return V.tolist()