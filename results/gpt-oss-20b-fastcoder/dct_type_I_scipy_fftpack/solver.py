from typing import Any
import numpy as np
from numpy.typing import NDArray
import scipy.fft

class Solver:
    def solve(self, problem: NDArray) -> NDArray:
        """
        Compute the N‑dimensional DCT Type I using the fast scipy.fft backend.
        """
        # Ensure the input is a NumPy array with a supported dtype
        arr = np.asarray(problem, dtype=np.float64)
        return scipy.fft.dctn(arr, type=1)