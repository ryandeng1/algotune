from typing import Any
import scipy.fftpack
from numpy.typing import NDArray

class Solver:

    def solve(self, problem: NDArray) -> NDArray:
        """
        Compute the N-dimensional DST Type II using scipy.fftpack.
        """
        result = scipy.fftpack.dstn(problem, type=2)
        return result
