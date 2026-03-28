import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[list[float]]:
        X = np.asarray(problem["X"], dtype=float)
        # Center the data
        X -= X.mean(axis=0)
        # Compute SVD
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        n_components = problem["n_components"]
        # components_ are rows of Vt[:n_components]
        V = Vt[:n_components]
        return V