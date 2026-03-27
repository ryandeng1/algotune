from typing import Any, Tuple, List
import numpy as np
from numpy.typing import NDArray


class Solver:
    def solve(self, problem: NDArray) -> Tuple[List[float], List[List[float]]]:
        """
        Solve the eigenvalue problem for the given real symmetric matrix.
        The solution returned is a tuple (eigenvalues, eigenvectors) where:
          - eigenvalues is a list of floats sorted in descending order.
          - eigenvectors is a list of lists, where each inner list represents the
            corresponding eigenvector (normalized to have unit length), sorted
            correspondingly to the eigenvalues.
        """
        # Get eigenvalues/eigenvectors; eigenvalues ascending, columns are vectors.
        eigvals, eigvecs = np.linalg.eigh(problem)

        # Reverse to descending order.
        eigvals = eigvals[::-1]

        # Transpose to get rows as eigenvectors, then reverse columns to match eigvals.
        eigvecs = eigvecs[:, ::-1].T

        # Convert to native Python lists (fastest via tolist on the whole array).
        return (eigvals.tolist(), eigvecs.tolist())