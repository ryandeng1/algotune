from typing import Any
import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray) -> list[list[complex]]:
        eigenvalues, eigenvectors = np.linalg.eig(problem)
        pairs = list(zip(eigenvalues, eigenvectors.T))
        pairs.sort(key=lambda x: (x[0].real, x[0].imag), reverse=True)
        return [
            (vec / np.linalg.norm(vec) if np.linalg.norm(vec) > 1e-12 else vec).tolist()
            for eigval, vec in pairs
        ]