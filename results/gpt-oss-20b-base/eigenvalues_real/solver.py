import numpy as np
from numpy.typing import NDArray


class Solver:
    def solve(self, problem: NDArray) -> list[float]:
        """
        Solve the eigenvalues problem for the given symmetric matrix.
        The returned list contains eigenvalues in descending order.
        """
        # np.linalg.eigh returns eigenvalues sorted in ascending order.
        # Reversing the array gives descending order with minimal overhead.
        eigenvalues = np.linalg.eigh(problem)[0]
        return list(eigenvalues[::-1])