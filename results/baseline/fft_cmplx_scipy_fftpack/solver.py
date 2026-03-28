from typing import Any
import scipy.fftpack as fftpack
from numpy.typing import NDArray

class Solver:

    def solve(self, problem: NDArray) -> NDArray:
        """
        Compute the N-dimensional FFT using scipy.fftpack.
        """
        return fftpack.fftn(problem)
