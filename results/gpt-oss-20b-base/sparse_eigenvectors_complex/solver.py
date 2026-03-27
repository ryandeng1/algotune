import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs


class Solver:
    def solve(self, problem: dict[str, Any]) -> list[complex]:
        """
        Solve the eigenvalue problem for the given square sparse matrix.
        The solution returned is a list of the eigenvectors with the largest `k`
        eigenvalues sorted in descending order by the modulus of the eigenvalue.
        """
        A = problem["matrix"]
        k = problem["k"]
        n, _ = A.shape

        # Ensure the matrix is in CSR format for faster computations
        if not sparse.isspmatrix_csr(A):
            A = A.tocsr()

        # Use a consistent deterministic start vector
        v0 = np.ones(n, dtype=A.dtype)

        # Compute the k largest magnitude eigenpairs
        eigvals, eigvecs = eigs(
            A,
            k=k,
            v0=v0,
            maxiter=n * 200,
            ncv=max(2 * k + 1, 20),
            return_eigenvectors=True,
        )

        # Sort eigenpairs by descending modulus of eigenvalue
        idx = np.argsort(-np.abs(eigvals))
        eigvecs = eigvecs[:, idx]

        # Return eigenvectors as a list of 1‑D arrays
        return [eigvecs[:, i] for i in range(k)]