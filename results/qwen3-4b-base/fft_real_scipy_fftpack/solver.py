from typing import Any
import scipy.fft as fft
from numpy.typing import NDArray


class Solver:
    def solve(self, problem: NDArray) -> NDArray:
        return fft.fftn(problem)