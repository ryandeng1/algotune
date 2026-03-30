# solver.py

from typing import List, Tuple, Any
import numpy as np
from scipy.signal import upfirdn

class Solver:
    """
    Solver for upfirdn operations.

    The input is a list of tuples (h, x, up, down) where
    - ``h`` is a 1‑D FIR filter
    - ``x`` is the signal to be filtered
    - ``up`` is the up‑sampling factor
    - ``down`` is the down‑sampling factor

    The output is a list of 1‑D numpy arrays containing the filtered signals.
    """

    @staticmethod
    def _to_1d(arr: Any) -> np.ndarray:
        """Ensure that the input array is 1‑D and cast to float."""
        a = np.asarray(arr, dtype=float)
        if a.ndim != 1:
            a = a.ravel()
        return a

    def solve(self, problem: List[Tuple[Any, Any, int, int]]) -> List[np.ndarray]:
        """
        Apply upfirdn to each problem definition.

        Parameters
        ----------
        problem : list of tuples
            Each tuple contains (h, x, up, down).

        Returns
        -------
        list of numpy.ndarray
            Results of the upfirdn operation for each tuple.
        """
        results: List[np.ndarray] = []

        for h, x, up, down in problem:
            # convert inputs to 1‑D numpy arrays – this removes the overhead of Python lists
            h_np = self._to_1d(h)
            x_np = self._to_1d(x)

            # vectorised upfirdn call – the real work is done in C/Fortran
            res = upfirdn(h_np, x_np, up=up, down=down)
            results.append(res)

        return results