import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray) -> list[complex]:
        eigenvalues = np.linalg.eigvals(problem)
        return sorted(eigenvalues, key=lambda x: (-x.real, -x.imag))
