from typing import Any
import scipy.fftpack
from numpy.typing import NDArray

class Solver:

    def solve(self, problem: NDArray) -> NDArray:
        """
        Compute the N-dimensional DCT Type I using scipy.fftpack.
        """
        result = scipy.fftpack.dctn(problem, type=1)
        return result
