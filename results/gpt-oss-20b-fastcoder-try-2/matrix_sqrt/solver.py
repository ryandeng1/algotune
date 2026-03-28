import numpy as np

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[complex]]]]:
        A = problem['matrix']
        try:
            # Eigenvalue decomposition (works for any square matrix)
            vals, vecs = np.linalg.eig(A)
            # Diagonal matrix of square roots of eigenvalues
            sqrt_vals = np.diag(np.sqrt(vals))
            # Compute the principal square root: X = V @ sqrt(D) @ V^{-1}
            X = vecs @ sqrt_vals @ np.linalg.inv(vecs)
        except Exception:
            return {'sqrtm': {'X': []}}
        return {'sqrtm': {'X': X.tolist()}}