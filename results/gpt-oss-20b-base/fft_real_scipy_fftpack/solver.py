import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray) -> NDArray:
        """
        Compute the N-dimensional FFT using NumPy's FFT implementation,
        which is faster and has lower overhead than SciPy's fftpack.
        """
        return np.fft.fftn(problem)