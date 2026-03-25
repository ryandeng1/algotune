import numpy as np
from numpy.typing import NDArray
from typing import List, Any


class Solver:
    def solve(self, problem: NDArray, **kwargs) -> List[complex]:
        """
        Compute the eigenvalues of a real square matrix and return them sorted
        in descending order by real part, then by imaginary part.

        Parameters
        ----------
        problem : NDArray
            2D NumPy array representing a real square matrix.

        Returns
        -------
        List[complex]
            List of the eigenvalues sorted as specified.
        """
        # Compute the eigenvalues
        eigenvalues = np.linalg.eig(problem)[0]

        # Sort: first by real part descending, then by imaginary part descending
        sorted_eigenvalues = sorted(eigenvalues, key=lambda z: (-z.real, -z.imag))

        return [complex(z) for z in sorted_eigenvalues]
