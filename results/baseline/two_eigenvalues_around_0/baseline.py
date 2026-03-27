from typing import Any
import numpy as np


class Solver:
    def solve(self, problem: dict[str, list[list[float]]]) -> list[float]:
        """
        Solve the problem by finding the two eigenvalues closest to zero.

        Args:
            problem (dict): Contains 'matrix', the symmetric matrix.

        Returns:
            list: The two eigenvalues closest to zero sorted by absolute value.
        """
        matrix = np.array(problem["matrix"], dtype=float)
        eigenvalues = np.linalg.eigvalsh(matrix)
        eigenvalues_sorted = sorted(eigenvalues, key=abs)
        return eigenvalues_sorted[:2]
