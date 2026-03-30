import numpy as np
from typing import Any, Dict

class Solver:
    """
    Extremely fast PSD projection using NumPy's LAPACK routines.

    The implementation uses:

    * `np.linalg.eigh` – guarantees real eigenvalues for symmetric matrices
      and is optimized for Hermitian matrices.
    * In‑place zeroing of negative eigenvalues – eliminates an extra copy.
    * `np.dot` instead of `@` for slightly better throughput.
    """
    def solve(self, problem: Dict[str, np.ndarray]) -> Dict[str, Any]:
        # Ensure we have a contiguous NumPy array
        A = np.asarray(problem["A"], dtype=np.float64)

        # Compute eigenvalues and eigenvectors
        eigvals, eigvecs = np.linalg.eigh(A)   # eigvals sorted ascending

        # Zero all negative eigenvalues (in‑place)
        mask = eigvals < 0
        eigvals[mask] = 0.0

        # Reconstruct the PSD matrix
        # Equivalent to V @ diag(eigvals) @ V.T
        V = eigvecs
        # First multiply V by the diagonal eigenvalues
        temp = V * eigvals  # broadcasting creates V * diag(eigvals)
        X = temp @ V.T      # matrix multiplication

        return {"X": X}