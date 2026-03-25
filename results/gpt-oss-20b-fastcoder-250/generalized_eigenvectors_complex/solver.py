import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray
from typing import Any, List, Tuple, Iterable

class Solver:
    def solve(self, problem: Tuple[NDArray, NDArray], **kwargs) -> Tuple[List[complex], List[List[complex]]]:
        """
        Compute the generalized eigenvalues and eigenvectors for A·x = λ B·x.
        The eigenvalues are sorted in descending order (real part, then imaginary part).
        Eigenvectors are normalized to unit Euclidean norm.
        """
        A, B = problem

        # Scale matrices for numerical stability
        scale = np.sqrt(np.linalg.norm(B, ord='fro'))
        if scale == 0:
            scale = 1.0
        B_s = B / scale
        A_s = A / scale

        # Solve generalized eigenproblem
        vals, vecs = la.eig(A_s, B_s)

        n = A.shape[0]
        # Normalize eigenvectors
        for i in range(n):
            v = vecs[:, i]
            norm = np.linalg.norm(v)
            if norm > 1e-15:
                vecs[:, i] = v / norm

        # Pair and sort
        pairs: List[Tuple[complex, NDArray]] = [(vals[i], vecs[:, i]) for i in range(n)]
        pairs.sort(key=lambda p: (-p[0].real, -p[0].imag))
        sorted_vals, sorted_vecs = zip(*pairs)

        # Convert to Python lists
        eigenvalues: List[complex] = [complex(v) for v in sorted_vals]
        eigenvectors: List[List[complex]] = [list(v) for v in sorted_vecs]

        return (eigenvalues, eigenvectors)
