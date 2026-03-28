from typing import Any
import numpy as np
import scipy.fft
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray) -> NDArray:
        """
        Compute the N-dimensional DCT Type I using scipy.fft.
        This uses the faster FFTW-backed implementation available in scipy.fft.
        """
        # scipy.fft.dctn defaults to type 2; specify type 1 explicitly
        return scipy.fft.dctn(problem, type=1, norm=None)