import numpy as np
from scipy import sparse

class Solver:
    def solve(self, problem):
        """
        Solve the eigenvalue problem for the given square sparse matrix.
        Returns the eigenvectors associated with the `k` largest-magnitude
        eigenvalues, sorted by decreasing |eigenvalue|.
        """
        A = problem["matrix"]
        k = problem["k"]
        N = A.shape[0]

        # Use the default orthogonal initial guess and a reasonable ncv
        ncv = max(2 * k + 1, 20)
        # Run eigs with a modest iteration limit
        vals, vecs = sparse.linalg.eigs(
            A, k=k, v0=np.ones(N, dtype=A.dtype), ncv=ncv, maxiter=N * 200
        )

        # argsort by decreasing absolute eigenvalue
        idx = np.argsort(-np.abs(vals))
        # Reorder eigenvectors to match sorted eigenvalues
        return [vecs[:, i] for i in idx]