import numpy as np
from scipy import fft
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray) -> NDArray:
        """
        Compute the N-dimensional DST Type II using scipy.fft which delegates
        to the fastest available implementation (pyfftw or FFTW).
        """
        # Ensure we use a float dtype that matches the problem
        arr = np.asarray(problem, dtype=float)
        return fft.dstn(arr, type=2)