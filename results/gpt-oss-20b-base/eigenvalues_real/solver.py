from numpy.typing import NDArray
import numpy as np

class Solver:
    """
    Solver for symmetric eigenvalue problems.
    """

    @staticmethod
    def solve(problem: NDArray) -> list[float]:
        """
        Return the eigenvalues of a symmetric matrix in descending order.

        Parameters
        ----------
        problem : NDArray
            Symmetric numpy matrix.

        Returns
        -------
        list[float]
            Eigenvalues sorted from largest to smallest.
        """
        # eigvalsh returns eigenvalues in ascending order for symmetric/hermitian matrices.
        return np.linalg.eigvalsh(problem)[::-1].tolist()