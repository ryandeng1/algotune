import numpy as np
from scipy import sparse, sparse.linalg

class Solver:
    """
    Fast eigenvector solver for sparse matrices.
    """
    def solve(self, problem: dict[str, Any]) -> list[complex]:
        A = problem['matrix']
        k = problem['k']
        N = A.shape[0]

        # Initial guess close to 1 in magnitude to speed up convergence
        v0 = np.full(N, 1, dtype=A.dtype)

        # eigs is efficient for large sparse problems.  We give it a tight
        # convergence budget that is proportional to the requested number of
        # eigenvalues to avoid unnecessary work.
        eigvals, eigvecs = sparse.linalg.eigs(
            A, k=k, v0=v0,
            maxiter=200 * k,          # a heuristic upper bound
            ncv=max(2 * k, 20),      # number of Lanczos vectors
            return_eigenvectors=True
        )

        # Sort by magnitude of eigenvalue and return the eigenvectors
        sorted_idx = np.argsort(-np.abs(eigvals))
        return [eigvecs[:, i] for i in sorted_idx]