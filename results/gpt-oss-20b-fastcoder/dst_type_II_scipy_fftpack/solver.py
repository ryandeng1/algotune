import numpy as np
from scipy.fft import dstn
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray) -> NDArray:
        """Compute the N‑dimensional Discrete Sine Transform Type‑II efficiently."""
        # Use SciPy's fast FFT implementation (Cython + FFTW if available)
        return dstn(problem, type=2)