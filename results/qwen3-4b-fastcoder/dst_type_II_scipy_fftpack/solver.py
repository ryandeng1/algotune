from typing import Any
import scipy.fft
from numpy.typing import NDArray


class Solver:
    def solve(self, problem: NDArray) -> NDArray:
        result = scipy.fft.dstn(problem, type=2)
        return result