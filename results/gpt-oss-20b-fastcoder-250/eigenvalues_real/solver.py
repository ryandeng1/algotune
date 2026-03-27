import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray) -> list[float]:
        """
        Solve the eigenvalues problem for the given symmetric matrix.
        The returned list contains the eigenvalues in descending order.
        """
        # np.linalg.eigvalsh returns eigenvalues of a Hermitian (symmetric) matrix in ascending order
        eigenvalues = np.linalg.eigvalsh(problem)
        # Reverse to descending order
        return eigenvalues[::-1].tolist()