from typing import Any
import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray) -> list[float]:
        eigenvalues = np.linalg.eigh(problem)[0]
        return eigenvalues[::-1].tolist()