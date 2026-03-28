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
        # eigvalsh is faster than eigh when only eigenvalues are needed
        ev = np.linalg.eigvalsh(problem)   # ascending order
        # reverse for descending order
        return ev[::-1].tolist()