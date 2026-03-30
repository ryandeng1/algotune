from numpy.typing import NDArray
import numpy as np

class Solver:
    """
    The Solver class implements an efficient eigenvalue solver for real square
    matrices.  It uses :func:`numpy.linalg.eigvals` to avoid computing the full
    eigendecomposition and sorts the eigenvalues with a single call to
    :func:`numpy.lexsort`.
    """

    @staticmethod
    def solve(problem: NDArray) -> list[complex]:
        """
        Compute all eigenvalues of *problem* and return them sorted in descending
        order using the key

            1. real part (descending)
            2. imaginary part (descending)

        Parameters
        ----------
        problem : NDArray
            Real square matrix whose eigenvalues are to be computed.

        Returns
        -------
        list[complex]
            Sorted list of eigenvalues.
        """
        # Fast single‑call eigvals; avoids matrix triangularisation if not needed
        vals = np.linalg.eigvals(problem)

        # Use lexsort on the negated components so that it yields a descending
        # order without a custom Python key function.
        order = np.lexsort((-vals.imag, -vals.real))
        sorted_vals = vals[order].tolist()
        return sorted_vals