from typing import Any
import numpy as np
from numpy.typing import NDArray


class Solver:
    def solve(self, problem: NDArray) -> tuple[list[float], list[list[float]]]:
        eigenvalues, eigenvectors = np.linalg.eigh(problem)
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]
        return (eigenvalues.tolist(), eigenvectors.T.tolist())