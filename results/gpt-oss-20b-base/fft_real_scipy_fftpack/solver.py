from typing import Any
import numpy as np
from numpy.typing import NDArray

class Solver:

    def solve(self, problem: NDArray) -> NDArray:
        """
        Compute the N-dimensional FFT using NumPy's optimized FFT implementation.
        """
        # Use NumPy's fftn which is faster than scipy.fftpack on modern releases.
        return np.fft.fftn(problem)