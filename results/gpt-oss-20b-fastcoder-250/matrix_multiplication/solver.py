# solver.py

import numpy as np
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, List[List[float]]], **kwargs) -> Any:
        """
        Compute the product of two matrices A and B provided in the `problem` dictionary.

        Parameters
        ----------
        problem : dict
            Dictionary with keys "A" and "B" containing 2‑D lists or arrays.

        Returns
        -------
        list[list[float]]
            The matrix product C = A · B as a nested list.
        """
        # Convert input to NumPy arrays (efficiently handles lists/arrays)
        A = np.array(problem["A"], dtype=np.float64, copy=False)
        B = np.array(problem["B"], dtype=np.float64, copy=False)

        # Multiply using BLAS via np.dot, which is highly optimised
        C = np.dot(A, B)

        # Convert back to nested Python lists for the expected output format
        return C.tolist()
