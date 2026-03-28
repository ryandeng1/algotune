from typing import Any
import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray) -> tuple[list[float], list[list[float]]]:
        """
        Solve the eigenvalue problem for the given real symmetric matrix.
        The solution returned is a tuple (eigenvalues, eigenvectors) where:
          - eigenvalues is a list of floats sorted in descending order.
          - eigenvectors is a list of lists, where each inner list represents the corresponding
            eigenvector (normalized to have unit length), sorted corresponding to the eigenvalues.

        :param problem: A numpy array representing the real symmetric matrix.
        :return: Tuple (eigenvalues, eigenvectors)
        """
        # Compute eigenvalues and eigenvectors (ascending order)
        vals, vecs = np.linalg.eigh(problem)

        # Reverse to get descending order
        vals = vals[::-1]
        vecs = vecs[:, ::-1]

        # Convert to Python lists
        return (vals.tolist(), vecs.T.tolist())