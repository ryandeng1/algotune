import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray
from typing import Tuple, List, Any

class Solver:
    def solve(self, problem: Tuple[NDArray, NDArray]) -> Tuple[List[complex], List[List[complex]]]:
        """
        Solve a generalized eigenvalue problem A·x = λ B·x.  Returns the
        eigenvalues sorted in descending order (by real part then imag part)
        together with the corresponding unit‑norm (reciprocal) eigenvectors.
        """
        A, B = problem
        # Scale the matrices only if the norm differs a lot from 1
        scale = np.linalg.norm(B, ord="fro")
        if scale != 0 and abs(scale - 1) > 1e-3:
            A = A / scale
            B = B / scale

        # The eigh interface solves the solved problem for symmetric/Hermitian matrices.
        # It returns the eigenvalues in ascending order.
        vals, vecs = la.eigh(A, B)

        # Normalize eigenvectors (reciprocal normalization is already performed by eigh)
        norms = np.linalg.norm(vecs, axis=0, keepdims=True)
        vecs = vecs / norms

        # Reverse order to obtain descending eigenvalues
        vals = vals[::-1]
        vecs = vecs[:, ::-1]

        # Sort by real part, then imaginary part, both descending
        order = np.lexsort((-vals.imag, -vals.real))
        vals = vals[order]
        vecs = vecs[:, order]

        # Convert to plain Python lists
        eigenvalues: List[complex] = vals.tolist()
        eigenvectors: List[List[complex]] = [vec.tolist() for vec in vecs.T]

        return eigenvalues, eigenvectors