import numpy as np
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, List[List[float]]], **kwargs) -> Any:
        """
        Compute the product of two matrices A and B from the problem dictionary.

        Parameters
        ----------
        problem : dict
            Dictionary with keys "A" and "B" whose values are 2‑dimensional lists of floats.
        **kwargs : any
            Unused keyword arguments.

        Returns
        -------
        numpy.ndarray
            The matrix product C = A · B as a NumPy array.
        """
        # Convert input lists to NumPy arrays and perform matrix multiplication
        A = np.array(problem["A"])
        B = np.array(problem["B"])
        return np.dot(A, B)
