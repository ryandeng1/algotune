import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: tuple[NDArray, NDArray]) -> list[float]:
        """
        Solve the generalized symmetric eigenvalue problem A·x = λ B·x.
        Uses NumPy's efficient generalized eigenvalue routine, then returns
        eigenvalues in descending order.
        """
        A, B = problem
        # Compute eigenvalues for the generalized symmetric-definite problem
        eigenvalues, _ = np.linalg.eigh(A, B)
        # Return sorted eigenvalues in descending order
        return list(reversed(eigenvalues.tolist()))