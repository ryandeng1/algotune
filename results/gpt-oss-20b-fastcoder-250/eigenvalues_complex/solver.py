import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray) -> list[complex]:
        """
        Solve the eigenvalue problem for a square real matrix.
        Returns a list of eigenvalues sorted in descending order by real part,
        breaking ties with the imaginary part.
        """
        # Efficient eigenvalue computation
        eig_vals = np.linalg.eigvals(problem)

        # Sort: primary key -real, secondary key -imag
        # Use lexsort with reversed order (higher first)
        # np.lexsort requires keys from least significant to most.
        key_real = -eig_vals.real
        key_imag = -eig_vals.imag
        idx = np.lexsort((key_imag, key_real))
        sorted_vals = eig_vals[idx]

        # Convert to Python list of complex
        return sorted_vals.tolist()