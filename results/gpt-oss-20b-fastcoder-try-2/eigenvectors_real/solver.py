import numpy as np
from numpy.typing import NDArray
from typing import Any, Tuple, List

class Solver:
    def solve(self, problem: NDArray) -> Tuple[List[float], List[List[float]]]:
        """
        Solve the eigenvalue problem for the given real symmetric matrix.
        The solution returned is a tuple (eigenvalues, eigenvectors) where:
          - eigenvalues is a list of floats sorted in descending order.
          - eigenvectors is a list of lists, each inner list representing
            the corresponding eigenvector (normalized to unit length),
            sorted to match the eigenvalues.

        :param problem: A numpy array representing the real symmetric matrix.
        :return: Tuple (eigenvalues, eigenvectors)
        """
        # Compute eigenpairs in ascending order and reverse to descending
        eigvals, eigvecs = np.linalg.eigh(problem)
        eigvals = eigvals[::-1].tolist()
        # Reverse columns to match eigenvalue order, then transpose to get list of rows
        eigvecs = eigvecs[:, ::-1].T.tolist()
        return eigvals, eigvecs