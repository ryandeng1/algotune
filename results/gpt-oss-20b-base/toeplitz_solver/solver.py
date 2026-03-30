# solver.py
import numpy as np
from scipy.linalg import solve_toeplitz
from typing import List, Dict

# Cache the relevant NumPy overloads a bit to avoid look‑ups during the call
_arange = np.arange
_astype = np.asarray

class Solver:
    """
    Fast solver for Toeplitz systems using SciPy's optimized routine.
    """

    def solve(self, problem: Dict[str, List[float]]) -> List[float]:
        """
        Solve the linear system Tx = b where T is Toeplitz, given by the first
        row `r` and first column `c`.  The input lists are converted to
        contiguous float64 NumPy arrays exactly once.

        Parameters
        ----------
        problem : dict
            Must contain keys 'c', 'r', and 'b' each mapping to a list of
            floats.

        Returns
        -------
        list[float]
            The solution vector x.
        """
        # Convert the lists to contiguous float64 arrays in one shot
        c = _astype(problem["c"], dtype=np.float64)
        r = _astype(problem["r"], dtype=np.float64)
        b = _astype(problem["b"], dtype=np.float64)

        # SciPy's solve_toeplitz expects a tuple of (c, r) ~ first column and row
        x = solve_toeplitz((c, r), b)
        return x.tolist()