from typing import Any
import numpy as np
from numpy.typing import NDArray


class Solver:
    def solve(self, problem: NDArray) -> NDArray:
        """
        Compute the N-dimensional FFT using NumPy's FFT implementation,
        which is typically faster and has fewer dependencies than
        scipy.fftpack.fftn.
        """
        return np.fft.fftn(problem)