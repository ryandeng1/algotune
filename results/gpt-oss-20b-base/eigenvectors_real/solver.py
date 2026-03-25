# solver.py
import numpy as np
from numpy.typing import NDArray
from typing import Any, List, Tuple

class Solver:
    def solve(self, problem: NDArray, **kwargs) -> Tuple[List[float], List[List[float]]]:
        """
        Compute eigenvalues and orthonormal eigenvectors of a real symmetric matrix.
        The eigenvalues are returned in descending order and each eigenvector is
        normalized to unit length.

        Parameters
        ----------
        problem : NDArray
            Real symmetric matrix (n x n).

        Returns
        -------
        Tuple[List[float], List[List[float]]]
            A tuple containing:
                - A list of eigenvalues sorted in descending order.
                - A list of corresponding eigenvectors, each a list of length n.
        """
        # np.linalg.eigh returns eigenvalues in ascending order and eigenvectors
        # as columns. Reverse both to get descending order.
        eigvals, eigvecs = np.linalg.eigh(problem)
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]

        # Convert to Python lists.  For eigenvectors we output row‑wise lists
        # (each row is one eigenvector) to match the specification.
        eigvals_list = eigvals.tolist()
        eigvecs_list = [eigvecs[i, :].tolist() for i in range(eigvecs.shape[0])]

        return eigvals_list, eigvecs_list
