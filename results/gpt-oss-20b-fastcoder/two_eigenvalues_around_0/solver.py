from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: dict[str, list[list[float]]]) -> list[float]:
        """
        Find the two eigenvalues of a symmetric matrix that are closest to zero.

        The implementation uses NumPy's efficient eigenvalue routine for Hermitian matrices
        and a two‑step partial sort (np.partition) to avoid sorting the entire array,
        which gives a noticeable speed gain for large matrices.
        """
        # Convert input matrix to a NumPy array of type float
        mat = np.array(problem['matrix'], dtype=float, copy=False)

        # Compute all eigenvalues of a symmetric matrix (Hermitian)
        eigs = np.linalg.eigvalsh(mat)

        # Find indices of the two eigenvalues with the smallest absolute value
        # using a partial sort: np.partition partially sorts the array so that
        # the first `k` elements are the smallest `k` (though not sorted themselves).
        k = 2
        abs_eigs = np.abs(eigs)
        indices = np.argpartition(abs_eigs, k-1)[:k]
        # Extract the selected eigenvalues
        selected = eigs[indices]
        # Sort them by absolute value to meet the specification
        selected_sorted = sorted(selected, key=abs)
        return selected_sorted