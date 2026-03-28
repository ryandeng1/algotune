from typing import Any
import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray) -> list[complex]:
        """
        Solve the eigenvalue problem for the given square matrix.
        The solution returned is a list of eigenvalues sorted in descending order.
        The sorting order is defined as: first by the real part (descending),
        then by the imaginary part (descending).
        """
        # Compute eigenvalues efficiently
        eigvals = np.linalg.eigvals(problem)

        # Create a structured array for efficient sorting
        dtype = np.dtype([('re', np.float64), ('im', np.float64), ('val', np.complex128)])
        structured = np.array(
            [(c.real, c.imag, c) for c in eigvals],
            dtype=dtype
        )

        # Sort by real part descending, then imaginary part descending
        sorted_idx = np.argsort(
            structured[['re', 'im']].view([('', np.float64)])[:, ::-1],
            kind='stable'
        )

        # Extract sorted eigenvalues
        sorted_eigvals = [structured['val'][i] for i in sorted_idx]
        return sorted_eigvals