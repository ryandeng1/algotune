import numpy as np
from numpy.typing import NDArray
from scipy import fft

class Solver:
    def solve(self, problem: NDArray) -> NDArray:
        """
        Compute the N-dimensional DCT Type I using the faster scipy.fft module.
        """
        return fft.dctn(problem, type=1, norm=None)