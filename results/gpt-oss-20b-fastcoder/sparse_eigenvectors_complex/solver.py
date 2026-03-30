# solver.py

import numpy as np
from scipy.sparse.linalg import eigs

class Solver:
    """
    Optimised eigenvalue solver.
    Uses SciPy's sparse eigs with vectorised sorting to minimise Python‑level overhead.
    """
    def solve(self, problem: dict) -> list[np.ndarray]:
        """
        Solve the eigenvalue problem for the given square sparse matrix.
        Returns the `k` eigenvectors with the largest eigenvalues (by absolute value)
        sorted from largest to smallest modulus.

        Parameters
        ----------
        problem : dict
            Dictionary with keys:
                "matrix" : sparse matrix (SciPy CSR/CSC)
                "k"      : int, number of eigenvalues/vectors to compute

        Returns
        -------
        list[ndarray]
            List of eigenvectors corresponding to the largest |eigenvalue|, in descending order.
        """
        A = problem["matrix"]
        k = int(problem["k"])

        N = A.shape[0]
        # Initial guess – all ones (matches data type of A)
        v0 = np.ones(N, dtype=A.dtype)

        # Compute the k dominant eigenpairs
        # maxiter and ncv are set to sensible defaults to avoid unnecessary work
        vals, vecs = eigs(A, k=k, v0=v0, maxiter=N * 200,
                           ncv=max(2 * k + 1, 20))

        # Sort indices by decreasing absolute value of eigenvalues
        idx = np.argsort(-np.abs(vals))

        # Rearrange eigenvectors according to sorted indices
        # vecs is shape (N, k); we transpose only once at the end
        sorted_vecs = vecs[:, idx].T

        # Convert to a Python list of 1‑D numpy arrays
        return [vec for vec in sorted_vecs]