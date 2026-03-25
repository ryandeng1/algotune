import numpy as np
from numpy.typing import NDArray
from typing import Any, List

class Solver:
    def solve(self, problem: NDArray, **kwargs) -> List[float]:
        """
        Compute the eigenvalues of a real symmetric matrix in descending order.

        Parameters
        ----------
        problem : NDArray
            Symmetric real-valued square matrix.

        Returns
        -------
        List[float]
            Eigenvalues sorted in descending order.
        """
        # Compute all eigenvalues efficiently for a symmetric matrix
        eigs = np.linalg.eigh(problem)[0]
        # Reverse for descending order and convert to a plain Python list
        return eigs[::-1].tolist()
