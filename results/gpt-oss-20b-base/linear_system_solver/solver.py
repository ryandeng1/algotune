import numpy as np
from typing import Any, List, Dict

class Solver:
    """
    Optimised solver for linear systems Ax = b.
    Uses NumPy's highly‑optimised ``linalg.solve`` routine with
    explicit dtype handling and contiguous arrays for the fastest data
    layout.  The function operates in O(n³) time, which is optimal for a
    dense linear‑system solver on platforms that expose BLAS/LAPACK.
    """

    @staticmethod
    def solve(problem: Dict[str, Any]) -> List[float]:
        """
        Solve the linear system Ax = b.

        Parameters
        ----------
        problem : dict
            A dictionary containing:
                - "A": 2‑D array‑like matrix of shape (n, n)
                - "b": 1‑D array‑like vector of length n

        Returns
        -------
        list[float]
            The solution vector x.
        """
        # Ensure we have NumPy arrays, avoid unnecessary copies
        A = np.asarray(problem["A"], dtype=np.float64, order="C")
        b = np.asarray(problem["b"], dtype=np.float64, order="C")

        # Directly call the fast LAPACK routine via linalg.solve
        x = np.linalg.solve(A, b)

        # Return a plain Python list of floats
        return x.tolist()