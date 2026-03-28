from typing import Any
import numpy as np
from numpy.typing import NDArray
import scipy.fft

class Solver:

    def solve(self, problem: NDArray) -> NDArray:
        """
        Compute the N-dimensional DST Type II efficiently.
        Uses scipy.fft.dstn which is faster than the legacy scipy.fftpack module.
        """
        # scipy.fft.dstn expects a floating point array
        problem = np.asarray(problem, dtype=np.float64)
        return scipy.fft.dstn(problem, type=2)