# solver.py
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List

class Solver:
    """
    Solves an eigenvalue problem for a real symmetric matrix.
    Returns eigenvalues in descending order together with their eigenvectors
    (each normalised to unit length) sorted the same way.
    """

    def solve(self, problem: NDArray) -> Tuple[List[float], List[List[float]]]:
        """
        Compute real eigenvalues and eigenvectors of a symmetric matrix.

        Parameters
        ----------
        problem : NDArray
            Symmetric real matrix.

        Returns
        -------
        Tuple[List[float], List[List[float]]]
            First element is a list of eigenvalues sorted in decreasing order.
            Second element is a list of the corresponding eigenvectors,
            each expressed as a list of floats.
        """
        # eigh returns eigenvalues in ascending order with columns as eigenvectors
        eig_vals, eig_vecs = np.linalg.eigh(problem)

        # reverse order to get descending eigenvalues
        eig_vals = eig_vals[::-1]
        eig_vecs = eig_vecs[:, ::-1]

        # Convert to Python lists efficiently
        eig_vals_list = eig_vals.tolist()
        eig_vecs_list = eig_vecs.T.tolist()  # rows correspond to vectors

        return eig_vals_list, eig_vecs_list