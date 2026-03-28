import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh, solve_triangular, cholesky

class Solver:
    def solve(self,
              problem: tuple[NDArray, NDArray]) -> tuple[list[float], list[list[float]]]:
        """
        Solve the generalized eigenvalue problem A·x = λ B·x for symmetric A and SPD B.
        Uses SciPy's `eigh` with the generalized algorithm, sorts eigenvalues in
        descending order and normalises the eigenvectors w.r.t. the B‑inner
        product.  The returned eigenvectors are column‑wise lists of floats.
        """
        A, B = problem

        # Solve the symmetric definite generalized eigenproblem directly.
        # `eigh` with a right-hand side returns eigenvalues in ascending order.
        evals, evecs = eigh(A, B, lower=True, check_finite=False)

        # Normalise eigenvectors with respect to B: vᵀBv = 1.
        # Use matrix multiplication with B once to avoid O(n²) loop.
        Bv = B @ evecs
        norms = np.sqrt(np.einsum('ij,ij->j', evecs, Bv))
        # avoid division by zero; norms should be >0 for SPD B
        evecs = evecs / norms

        # Reverse to descending order
        evals = evals[::-1]
        evecs = evecs[:, ::-1]

        return (evals.tolist(), evecs.T.tolist())