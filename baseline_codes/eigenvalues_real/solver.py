from typing import Any
import numpy as np
from numpy.typing import NDArray

class Solver:

    def solve(self, problem: NDArray) -> list[float]:
        """
        Solve the eigenvalues problem for the given symmetric matrix.
        The solution returned is a list of eigenvalues in descending order.

        :param problem: A symmetric numpy matrix.
        :return: List of eigenvalues in descending order.
        """
        eigenvalues = np.linalg.eigh(problem)[0]
        solution = sorted(eigenvalues, reverse=True)
        return solution
