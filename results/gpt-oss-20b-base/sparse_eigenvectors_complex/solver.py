# solver.py
from typing import Any, List
import numpy as np
from scipy import sparse

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> List[np.ndarray]:
        """
        Compute the eigenvectors corresponding to the k largest
        (in magnitude) eigenvalues of a square sparse matrix.

        Parameters
        ----------
        problem : dict
            Dictionary with keys:
                - "matrix": a scipy.sparse matrix (CSR or similar)
                - "k": number of eigenvectors to return (int)

        Returns
        -------
        List[np.ndarray]
            List of eigenvectors (complex arrays) sorted in descending
            order by the modulus of the corresponding eigenvalue.
        """
        A = problem["matrix"]
        k = problem["k"]

        # Ensure matrix is in CSR format for efficient eigenvalue computation
        if not sparse.isspmatrix_csr(A):
            A = A.tocsr()

        N = A.shape[0]

        # Deterministic starting vector for reproducibility
        v0 = np.ones(N, dtype=A.dtype)

        # Compute eigenvalues and eigenvectors using the sparse eigen solver
        eigenvalues, eigenvectors = sparse.linalg.eigs(
            A,
            k=k,
            v0=v0,
            maxiter=N * 200,
            ncv=max(2 * k + 1, 20),
            return_eigenvectors=True,
        )

        # Pair eigenvalues with their corresponding eigenvectors
        pairs = list(zip(eigenvalues, eigenvectors.T))

        # Sort pairs by descending absolute value of eigenvalue
        pairs.sort(key=lambda pair: -np.abs(pair[0]))

        # Extract sorted eigenvectors
        solution = [pair[1] for pair in pairs]

        return solution
