from numpy.typing import NDArray
import numpy as np

class Solver:
    def solve(self, problem: NDArray) -> NDArray:
        """
        Compute the N-dimensional FFT efficiently using NumPy.
        """
        return np.fft.fftn(problem)