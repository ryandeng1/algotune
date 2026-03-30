from typing import Any, List
import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray) -> List[float]:
        """
        Solve the eigenvalues problem for the given symmetric matrix.
        The solution returned is a list of eigenvalues in descending order.

        :param problem: A symmetric numpy matrix.
        :return: List of eigenvalues in descending order.
        """
        # eigvalsh is optimised for symmetric/hermitian matrices and returns
        # eigenvalues sorted in ascending order. We simply reverse the order
        # to get a descending list, avoiding a costly Python sort.
        return np.linalg.eigvalsh(problem)[::-1].tolist()