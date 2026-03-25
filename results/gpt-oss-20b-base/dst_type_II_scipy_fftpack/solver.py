import numpy as np
from scipy.fft import dstn

class Solver:
    def solve(self, problem: np.ndarray) -> np.ndarray:
        """
        Compute the 2‑D Discrete Sine Transform (DST) Type II of the input array.

        The implementation leverages SciPy's fast Fourier transform routine
        which is highly optimised in C/Cython.  It operates directly on the
        entire n×n array and returns the real‑valued frequency components
        with numerical precision matching the reference implementation.
        """
        # Ensure the input is a float array for numerical stability
        arr = np.asarray(problem, dtype=np.float64)
        # Compute the N‑dimensional DST Type II
        return dstn(arr, type=2, norm=None)
