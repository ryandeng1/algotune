import numpy as np
from typing import Any


class Solver:
    def solve(self, problem: dict[str, list[list[float]]]) -> list[float]:
        """
        Return the two eigenvalues of the given symmetric matrix that are
        closest to zero (in absolute value).
        """
        # Convert the input to a NumPy array only once
        matrix = np.asarray(problem["matrix"], dtype=float)

        # `eigvalsh` returns the eigenvalues of a Hermitian (symmetric) matrix
        # in ascending order (by value).  We need the two with smallest |λ|.
        eig = np.linalg.eigvalsh(matrix)

        # Find indices of the two smallest absolute eigenvalues using
        # `argpartition` for O(n) average time.
        idx = np.argpartition(np.abs(eig), 2)[:2]

        # Sort the selected eigenvalues by their absolute value for a defined order.
        result = np.take(eig, idx)
        result.sort(key=abs)
        return result.tolist()