import numpy as np
from typing import List
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray) -> List[float]:
        """
        Solve the eigenvalues problem for the given symmetric matrix.
        The solution returned is a list of eigenvalues in descending order.
        """
        # Compute eigenvalues of a symmetric (Hermitian) matrix
        eigvals = np.linalg.eigvalsh(problem)
        # Sort in descending order
        eigvals_desc = np.sort(eigvals)[::-1]
        # Convert to plain Python list for the return type
        return eigvals_desc.tolist()