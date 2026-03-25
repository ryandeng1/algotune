import numpy as np
from numpy.typing import NDArray
from typing import Any, List, Tuple

class Solver:
    def solve(self, problem: NDArray, **kwargs) -> Tuple[List[float], List[List[float]]]:
        """
        Compute eigenvalues and orthonormal eigenvectors of a real symmetric matrix.
        The eigenvalues are sorted in descending order and eigenvectors are rows
        matching the eigenvalues.
        """
        # Use NumPy's efficient symmetric solver
        eigenvalues, eigenvectors = np.linalg.eigh(problem)  # eigenvectors as columns
        # Reverse to get descending order
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]
        # Convert to Python lists with float precision
        eigenvalues_list = eigenvalues.tolist()
        # Transpose to get rows as eigenvectors
        eigenvectors_list = [list(row) for row in eigenvectors.T]
        return eigenvalues_list, eigenvectors_list
