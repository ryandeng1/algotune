from typing import Any
import numpy as np
from numpy.typing import NDArray


class Solver:
    def solve(self, problem: NDArray) -> list[list[complex]]:
        eigenvalues, eigenvectors = np.linalg.eig(problem)
        pairs = list(zip(eigenvalues, eigenvectors.T))
        pairs.sort(key=lambda pair: (-pair[0].real, -pair[0].imag))
        sorted_evecs = []
        for eigval, vec in pairs:
            norm = np.linalg.norm(vec)
            if norm > 1e-12:
                vec /= norm
            sorted_evecs.append(vec.tolist())
        return sorted_evecs