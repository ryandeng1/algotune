import numpy as np
from scipy import sparse

class Solver:
    def solve(self, problem: dict[str, any]) -> list[complex]:
        """
        Solve the eigenvalue problem for a sparse square matrix.
        The output is a list of the eigenvectors associated with the largest
        k eigenvalues, sorted in descending order of eigenvalue modulus.
        """
        A = problem["matrix"]
        k = problem["k"]
        N = A.shape[0]

        # Deterministic starting vector
        v0 = np.ones(N, dtype=A.dtype)

        # Compute the k eigenvalues/vectors
        eigvals, eigvecs = sparse.linalg.eigs(
            A,
            k=k,
            v0=v0,
            maxiter=N * 200,
            ncv=max(2 * k + 1, 20),
        )

        # Order indices by decreasing eigenvalue magnitude
        idx = np.argsort(-np.abs(eigvals))

        # Reorder the eigenvectors accordingly
        return [eigvecs[:, i] for i in idx]