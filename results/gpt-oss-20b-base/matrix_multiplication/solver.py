from __future__ import annotations
from typing import Any, List, Dict
import numpy as np

class Solver:
    """
    Solver for fast matrix multiplication.
    The implementation leverages NumPy's highly optimised BLAS backend,
    which is the fastest available for large dense matrices in pure Python.
    """

    def solve(self, problem: Dict[str, List[List[float]]]) -> List[List[float]]:
        """
        Multiply matrices A and B from the problem dict and return the result
        as a Python list of lists.

        Parameters
        ----------
        problem : dict
            Must contain keys 'A' and 'B' with values being 2‑D lists of floats
            representing the input matrices.

        Returns
        -------
        list[list[float]]
            The product matrix C = A @ B.
        """
        # Convert inputs to NumPy arrays with the best possible dtype.
        # Using float64 ensures compatibility with BLAS.
        A = np.array(problem['A'], dtype=np.float64, copy=False)
        B = np.array(problem['B'], dtype=np.float64, copy=False)

        # Perform matrix multiplication; np.dot uses BLAS when available.
        C = np.dot(A, B)

        # Return as a plain Python list of lists for the required API.
        return C.tolist()