import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray) -> NDArray:
        """Compute the N‑dimensional FFT using NumPy's highly optimized routine."""
        return np.fft.fftn(problem)