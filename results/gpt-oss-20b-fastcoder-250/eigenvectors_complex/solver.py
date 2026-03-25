import numpy as np
from typing import Any, List, Tuple


class Solver:
    def solve(self, problem: np.ndarray, **kwargs) -> Tuple[List[complex], List[List[complex]]]:
        """
        Compute the eigenvalues and eigenvectors of a real square matrix.
        The eigenpairs are sorted in descending order by the eigenvalue's real part
        and then by its imaginary part. Eigenvectors are returned as lists of
        complex numbers and are normalized to unit Euclidean norm.

        Parameters
        ----------
        problem : np.ndarray
            A non‑symmetric real square matrix.

        Returns
        -------
        Tuple[List[complex], List[List[complex]]]
            First entry: sorted list of eigenvalues.
            Second entry: list of corresponding normalized eigenvectors.
        """
        # Compute raw eigenpairs using NumPy
        eig_vals, eig_vecs = np.linalg.eig(problem)

        # Zip, sort, and normalize
        pairs = list(zip(eig_vals, eig_vecs.T))
        pairs.sort(key=lambda p: (-p[0].real, -p[0].imag))

        sorted_eigs = []
        sorted_vecs = []

        for val, vec in pairs:
            sorted_eigs.append(val)
            vec = np.asarray(vec, dtype=complex)
            norm = np.linalg.norm(vec)
            if norm == 0:
                # Numerical safeguard – keep zero vector as is
                vec = vec
            else:
                vec = vec / norm
            sorted_vecs.append(vec.tolist())

        return sorted_eigs, sorted_vecs
