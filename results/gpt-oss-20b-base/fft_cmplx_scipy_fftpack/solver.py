from typing import Any
import numpy as np
from numpy.typing import NDArray

class Solver:

    def solve(self, problem: NDArray) -> NDArray:
        """
        Compute the N‑dimensional FFT using NumPy, which is typically faster
        than scipy.fftpack for large arrays.
        """
        return np.fft.fftn(problem)