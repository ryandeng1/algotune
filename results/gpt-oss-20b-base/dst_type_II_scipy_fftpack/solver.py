from typing import Any
import numpy as np
import scipy.fft
from numpy.typing import NDArray

class Solver:
    """
    Compute the N-dimensional DST Type II using scipy.fft (faster implementation).
    """
    def solve(self, problem: NDArray) -> NDArray:
        # Use scipy.fft's dstn which dispatches to a highly-optimized routine
        # If available, it uses FFTW via pyfftw under the hood for speed.
        return scipy.fft.dstn(problem, type=2)