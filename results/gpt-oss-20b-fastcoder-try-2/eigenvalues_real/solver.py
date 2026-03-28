import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray) -> list[float]:
        """
        Solve the eigenvalues problem for the given symmetric matrix.
        The solution returned is a list of eigenvalues in descending order.
        """
        # eigh returns eigenvalues in ascending order for symmetric matrices.
        eigenvalues = np.linalg.eigh(problem)[0]
        # Reverse to get descending order without extra Python sort overhead.
        return eigenvalues[::-1].tolist()