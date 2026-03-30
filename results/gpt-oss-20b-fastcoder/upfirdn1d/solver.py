# solver.py

from typing import List, Tuple
import numpy as np
from scipy.signal import upfirdn

class Solver:
    """
    Efficient solver for multiple upfirdn problems.

    Each problem is defined as a tuple (h, x, up, down) where:
    - h    : FIR filter coefficients (1‑D array or list)
    - x    : Input sequence (1‑D array or list)
    - up   : Upsampling factor (int)
    - down : Downsampling factor (int)

    The solver returns a list of 1‑D numpy arrays containing the computed
    upfirdn results in the same order as the input list.
    """

    @staticmethod
    def _to_ndarray(arr: List[float]) -> np.ndarray:
        """Convert a list or array to a 1‑D numpy float64 array."""
        if isinstance(arr, np.ndarray):
            return arr.astype(np.float64, copy=False)
        return np.asarray(arr, dtype=np.float64)

    def solve(self, problem: List[Tuple[List[float], List[float], int, int]]) -> List[np.ndarray]:
        """
        Compute the upfirdn operation for each problem definition in the list.

        Parameters
        ----------
        problem : list of tuples
            Each tuple contains (h, x, up, down).

        Returns
        -------
        list of np.ndarray
            The upfirdn results for each input.
        """
        # Pre‑allocate result list
        results: List[np.ndarray] = [None] * len(problem)

        # Iterate using enumerate to avoid multiple list lookups
        for idx, (h, x, up, down) in enumerate(problem):
            # Ensure 1‑D float64 arrays for speed
            h_arr = self._to_ndarray(h)
            x_arr = self._to_ndarray(x)

            # Perform the upfirdn operation
            results[idx] = upfirdn(h_arr, x_arr, up=up, down=down)

        return results