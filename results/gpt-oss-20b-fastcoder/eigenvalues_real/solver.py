import numpy as np

class Solver:
    def solve(self, problem: np.ndarray) -> list[float]:
        """
        Return eigenvalues of a symmetric matrix in descending order.
        """
        # eigvalsh returns sorted ascending eigenvalues for Hermitian matrices
        vals = np.linalg.eigvalsh(problem)
        # reverse to descending order
        return vals[::-1].tolist()