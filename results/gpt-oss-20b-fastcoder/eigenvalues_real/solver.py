import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray) -> list[float]:
        """Return eigenvalues of a symmetric matrix in descending order."""
        # eigh returns eigenvalues in ascending order
        asc = np.linalg.eigh(problem)[0]
        # Reverse order to descending without an extra sort pass
        return list(asc[::-1])