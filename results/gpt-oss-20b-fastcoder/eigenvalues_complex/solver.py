import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray) -> list[complex]:
        """
        Solve the eigenvalue problem for the given square matrix.
        The solution returned is a list of eigenvalues sorted in descending order:
        first by the real part, then by the imaginary part.
        """
        # Fast eigenvalue computation
        eig = np.linalg.eigvals(problem)

        # Lexicographic descending sort: primary real, secondary imag
        # Use negative values so np.lexsort (ascending) gives descending order
        idx = np.lexsort(
            ( -eig.imag,          # secondary key
              -eig.real ) )      # primary key

        return eig[idx].tolist()