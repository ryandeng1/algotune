import numpy as np
from numpy.typing import NDArray
from typing import Any

class Solver:
    def solve(self, problem: NDArray) -> NDArray:
        """
        Compute the N-dimensional FFT using numpy.fft.fftn, which
        is generally faster than scipy.fftpack.fftn for pure
        NumPy arrays.
        """
        return np.fft.fftn(problem)