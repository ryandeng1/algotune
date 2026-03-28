from typing import Any
import numpy as np
from numpy.typing import NDArray


class Solver:
    def solve(self, problem: NDArray) -> list[complex]:
        eigenvalues = np.linalg.eig(problem)[0]
        sorted_indices = np.argsort((-eigenvalues.real, -eigenvalues.imag))
        return eigenvalues[sorted_indices].tolist()