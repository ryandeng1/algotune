from typing import Any
from numpy.typing import NDArray
import numpy as np
from numpy.fft import fftn

class Solver:
    def solve(self, problem: NDArray) -> NDArray:
        return fftn(problem)