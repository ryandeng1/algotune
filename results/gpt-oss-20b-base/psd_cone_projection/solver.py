import numpy as np

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Projects a symmetric matrix onto the positive semidefinite cone.
        This implementation uses the efficient eigh routine and avoids
        explicit construction of a diagonal matrix.
        """
        A = np.asarray(problem['A'])
        # Ensure the matrix is symmetric
        if not np.allclose(A, A.T, atol=1e-12):
            A = (A + A.T) / 2.0

        # Eigen‑decomposition for symmetric matrices
        eigvals, eigvecs = np.linalg.eigh(A)
        # Threshold negative eigenvalues to zero
        eigvals = np.clip(eigvals, 0, None)

        # Instead of diag(eigvals) use column‑wise scaling
        Y = eigvecs * eigvals
        X = Y @ eigvecs.T
        return {'X': X}