import numpy as np
from typing import Dict, List

class Solver:
    """Optimised solver for finding the two eigenvalues nearest zero."""

    @staticmethod
    def _eigvals_nearest_zero(mat: np.ndarray, k: int = 2) -> List[float]:
        """
        Compute the k eigenvalues of a symmetric matrix with smallest absolute values.
        Uses numpy's efficient *eigvalsh* routine and :func:`numpy.argpartition`.

        Parameters
        ----------
        mat : np.ndarray
            Symmetric input matrix.
        k : int, optional
            Number of eigenvalues to return (default: 2).

        Returns
        -------
        List[float]
            The k eigenvalues closest to zero, sorted by absolute value.
        """
        # Fast eigenvalue computation for symmetric matrices.
        eigs = np.linalg.eigvalsh(mat)          #  Rank‑ordered from small to large.
        # Find indices of the k smallest |eigenvalue| without a full sort.
        idx = np.argpartition(np.abs(eigs), k)[:k]
        # Extract, sort by absolute value, and convert to Python list.
        return sorted((eigs[i] for i in idx), key=abs)

    def solve(self, problem: Dict[str, List[List[float]]]) -> List[float]:
        matrix = np.array(problem["matrix"], dtype=np.float64, order="C")
        return self._eigvals_nearest_zero(matrix, k=2)