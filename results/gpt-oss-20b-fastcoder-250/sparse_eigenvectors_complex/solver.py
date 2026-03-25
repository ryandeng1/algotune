# solver.py
from typing import Any, List
import numpy as np
from scipy import sparse

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> List[np.ndarray]:
        """Compute the eigenvectors corresponding to the k eigenvalues
        with largest magnitude of a sparse complex matrix."""
        A: sparse.spmatrix = problem["matrix"]
        k: int = problem["k"]
        N: int = A.shape[0]

        # Deterministic starting vector for reproducibility
        v0 = np.ones(N, dtype=A.dtype)

        # Use ARPACK via scipy.sparse.linalg.eigs
        eigenvalues, eigenvectors = sparse.linalg.eigs(
            A,
            k=k,
            v0=v0,
            maxiter=N * 200,
            ncv=max(2 * k + 1, 20),
        )

        # Pair eigenvalues with eigenvectors and sort by magnitude
        pairs = list(zip(eigenvalues, eigenvectors.T))
        pairs.sort(key=lambda pair: -np.abs(pair[0]))

        # Extract sorted eigenvectors
        return [pair[1] for pair in pairs]
