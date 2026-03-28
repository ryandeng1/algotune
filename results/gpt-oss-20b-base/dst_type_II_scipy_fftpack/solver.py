from typing import Any
import scipy.fft  # faster and newer FFT implementation
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray) -> NDArray:
        """
        Compute the N-dimensional DST Type II using scipy.fft.
        """
        return scipy.fft.dstn(problem, type=2)