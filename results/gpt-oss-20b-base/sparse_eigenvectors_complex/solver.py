import numpy as np
from scipy import sparse

class Solver:
    def solve(self, problem: dict) -> list:
        """
        Return the eigenvectors corresponding to the `k` eigenvalues with
        largest absolute value, sorted in decreasing order.

        The matrix `A` is expected to be a square SciPy sparse matrix.
        """
        A = problem["matrix"]
        k = problem["k"]
        n = A.shape[0]

        # Use an initial guess that improves convergence speed
        v0 = np.ones(n, dtype=A.dtype)

        # ncv is the number of Lanczos vectors: keep it modest for speed
        ncv = max(2 * k + 1, 20)

        # Perform the eigenvalue solve
        vals, vecs = sparse.linalg.eigs(A, k=k, v0=v0, maxiter=n * 200,
                                        ncv=ncv, which="LM")

        # Sort by magnitude of eigenvalue (descending)
        idx = np.argsort(-np.abs(vals))
        return [vecs[:, i] for i in idx]