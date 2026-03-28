from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: dict[str, list[list[float]]]) -> list[float]:
        matrix = np.array(problem["matrix"], dtype=float)
        eigenvalues = np.linalg.eigvalsh(matrix)
        abs_eigen = np.abs(eigenvalues)
        indices = np.argpartition(abs_eigen, 2)
        return eigenvalues[indices[:2]].tolist()