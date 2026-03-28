import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray) -> list[complex]:
        """Return eigenvalues of a real square matrix sorted by real part (descending) and imaginary part (descending)."""
        # Fast eigenvalue computation: eigvals returns only eigenvalues.
        vals = np.linalg.eigvals(problem)

        # np.lexsort expects keys in ascending order, so we negate to get descending.
        # The first key is the most significant (real part), the second is the imaginary part.
        order = np.lexsort((-vals.imag, -vals.real))

        # Apply the ordering and convert to a plain Python list of complex numbers.
        return vals[order].tolist()