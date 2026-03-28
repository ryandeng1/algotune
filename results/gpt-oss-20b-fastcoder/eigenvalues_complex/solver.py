import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray) -> list[complex]:
        """
        Solve the eigenvalue problem for the given square matrix.

        The returned list contains the eigenvalues sorted in descending order
        according to real part first and then imaginary part.

        Parameters
        ----------
        problem : NDArray
            Square real matrix.

        Returns
        -------
        list[complex]
            Sorted eigenvalues.
        """
        # Compute only eigenvalues (no eigenvectors) for speed
        eigvals = np.linalg.eigvals(problem)

        # Sort by real part descending, then by imaginary part descending
        # np.lexsort sorts according to the last key first, hence the
        # negative signs to achieve descending order.
        order = np.lexsort((-eigvals.imag, -eigvals.real))
        sorted_vals = eigvals[order]

        return sorted_vals.tolist()