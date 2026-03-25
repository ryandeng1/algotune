import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[list[float]]:
        X = np.array(problem["X"])
        n_components = problem["n_components"]
        X_centered = X - np.mean(X, axis=0)
        
        U, S, Vh = np.linalg.svd(X_centered, full_matrices=False)
        
        V = Vh[:, :n_components].T
        return V.tolist()
