# solver.py
import numpy as np
from numpy.typing import NDArray
from typing import Any, List, Tuple, Iterable


class Solver:
    """
    Solver that computes eigenvalues and eigenvectors of a real square matrix.
    The eigenvalues are sorted descending by real part then imaginary part.
    Eigenvectors are column vectors of the eigenvectors matrix, normalized to unit
    Euclidean norm, and returned as lists of complex numbers.
    """

    @staticmethod
    def _sort_pairs(values: np.ndarray, vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sort eigenvalues and corresponding eigenvectors according to the specified order.
        """
        # Get sort indices: descending real, then descending imag
        # Using argsort on negative values ensures descending order
        sort_idx = np.lexsort(
            (
                -values.imag,
                -values.real,
            )
        )
        return values[sort_idx], vectors[:, sort_idx]

    def solve(self, problem: NDArray, **kwargs) -> List[List[complex]]:
        """
        Parameters
        ----------
        problem : NDArray
            Real square matrix.

        Returns
        -------
        List[List[complex]]
            List of normalized eigenvectors sorted according to the eigenvalues
            (sorted descending by real part, then imaginary part).
        """
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(problem)

        # Sort pairs
        eigenvalues, eigenvectors = self._sort_pairs(eigenvalues, eigenvectors)

        # Normalize eigenvectors and convert to list-of-lists
        norms = np.linalg.norm(eigenvectors, axis=0)
        # Avoid division by zero: very small vectors are set to zero already
        safe_norms = np.where(norms > 1e-12, norms, 1.0)
        normalized = eigenvectors / safe_norms

        # Convert to list of lists of complex numbers
        return [
            [complex(val.real, val.imag) for val in normalized[:, i]]
            for i in range(normalized.shape[1])
        ]
