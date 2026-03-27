from typing import Any
import numpy as np
from numpy.typing import NDArray


class Solver:
    def solve(self, problem: NDArray) -> NDArray:
        """
        Compute the N-dimensional FFT using NumPy's efficient implementation.
        """
        # Convert to a NumPy array if necessary (doesn’t copy if already an array)
        arr = np.asarray(problem)
        # Compute the N‑dimensional fast Fourier transform
        return np.fft.fftn(arr)