from typing import Any
import numpy as np
from numpy.typing import NDArray


class Solver:
    def solve(self, problem: NDArray) -> tuple[list[float], list[list[float]]]:
        eigenvalues, eigenvectors = np.linalg.eigh(problem)
        return (
            eigenvalues[::-1].tolist(),
            eigenvectors[:, ::-1].T.tolist()
        )