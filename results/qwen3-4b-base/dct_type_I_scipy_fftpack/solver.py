from typing import Any
import scipy.fftpack
from numpy.typing import NDArray


class Solver:
    def solve(self, problem: NDArray) -> NDArray:
        return scipy.fftpack.dctn(problem, type=1, overwrite_input=True)