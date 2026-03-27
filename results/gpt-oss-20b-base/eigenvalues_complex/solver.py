from typing import Any
import numpy as np
from numpy.typing import NDArray


class Solver:
    def solve(self, problem: NDArray) -> list[complex]:
        """
        Compute eigenvalues of the given square matrix and return them
        sorted descending by real part and then by imaginary part.
        """
        # Compute only eigenvalues for speed
        evs = np.linalg.eigvals(problem)

        # Vectorized sorting: lexsort uses last key as primary,
        # so we provide negative imag and negative real
        idx = np.lexsort((-evs.imag, -evs.real))
        return evs[idx].tolist()