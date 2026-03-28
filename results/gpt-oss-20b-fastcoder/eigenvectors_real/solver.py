import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray) -> tuple[list[float], list[list[float]]]:
        """
        Solve the eigenvalue problem for a real symmetric matrix.

        Returns eigenvalues sorted in descending order and their corresponding
        normalized eigenvectors consistently sorted.
        """
        # eigh returns values in ascending order
        eigenvalues, eigenvectors = np.linalg.eigh(problem)

        # Reverse both arrays to get descending order
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]

        # Convert to Python lists efficiently
        eigenvalues_list = eigenvalues.tolist()
        # Transpose once and then convert: shape (n, n) -> list of n lists
        eigenvectors_list = eigenvectors.T.tolist()

        return eigenvalues_list, eigenvectors_list