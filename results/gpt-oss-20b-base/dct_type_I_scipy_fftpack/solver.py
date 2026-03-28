from typing import Any
import numpy as np
from numpy.typing import NDArray
from scipy.fft import dctn

class Solver:
    def solve(self, problem: NDArray) -> NDArray:
        """
        Compute the N‑dimensional DCT Type I using scipy.fft (more efficient than fftpack).
        """
        return dctn(problem, type=1, norm=None)