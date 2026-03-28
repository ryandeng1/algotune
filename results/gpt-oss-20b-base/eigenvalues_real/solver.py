import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray) -> list[float]:
        """
        Solve the eigenvalues problem for the given symmetric matrix.
        Return a list of eigenvalues in descending order.
        """
        # np.linalg.eigvalsh is cheaper than a full eigh call
        vals = np.linalg.eigvalsh(problem)
        # fast sorting via numpy
        return list(np.sort(vals)[::-1])