import numpy as np
from typing import Any
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray) -> NDArray:
        """
        Compute the N-dimensional FFT using NumPy's highly optimised backend
        (which typically leverages FFTW or MKL under the hood).
        """
        return np.fft.fftn(problem)