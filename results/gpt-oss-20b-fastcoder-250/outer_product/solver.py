# solver.py
import numpy as np
from typing import Any, Tuple

class Solver:
    def solve(self, problem: Tuple[np.ndarray, np.ndarray], **kwargs) -> Any:
        """
        Compute the outer product of two vectors using NumPy's efficient routine.
        The input `problem` is a tuple (vec1, vec2) where both are 1-D arrays of the same length.

        Parameters
        ----------
        problem : Tuple[np.ndarray, np.ndarray]
            The two input vectors.

        Returns
        -------
        np.ndarray
            The n x n matrix containing the outer product of the vectors.
        """
        vec1, vec2 = problem
        # np.outer is already highly optimized in C and handles broadcasting
        return np.outer(vec1, vec2)
