from typing import Any, Tuple, List
import numpy as np
from numpy.typing import NDArray


class Solver:
    """
    Optimised solver for real symmetric eigenvalue problems.
    """

    def solve(self, problem: NDArray) -> Tuple[List[float], List[List[float]]]:
        """
        Compute eigenvalues and eigenvectors for a real symmetric matrix.

        Parameters
        ----------
        problem : NDArray
            Real symmetric matrix.

        Returns
        -------
        Tuple[List[float], List[List[float]]]
            Eigenvalues sorted in descending order and corresponding
            eigenvectors (unit‑norm) also sorted accordingly.
        """
        # NumPy already uses LAPACK's Eigh for symmetric matrices → very fast.
        eigvals, eigvecs = np.linalg.eigh(problem)

        # Reverse to descending order
        eigvals = np.flip(eigvals)
        eigvecs = eigvecs[:, ::-1]

        # Convert to Python lists in a single pass
        eigvals_list = eigvals.tolist()
        eigvecs_list = eigvecs.T.tolist()   # transpose once, then list

        return eigvals_list, eigvecs_list