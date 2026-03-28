from typing import Any
import scipy.fftpack
from numpy.typing import NDArray


class Solver:
    def solve(self, problem: NDArray) -> NDArray:
        result = scipy.fftpack.dstn(problem)
        return result