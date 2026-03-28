import numpy as np
from scipy.fft import dctn
from numpy.typing import NDArray
from typing import Any

class Solver:
    def solve(self, problem: NDArray) -> NDArray:
        """
        Compute the N‑dimensional DCT Type I using SciPy's fast FFT implementation.
        """
        # Use scipy.fft.dctn which is backed by FFTW (or MKL) and is faster than fftpack.
        return dctn(problem, type=1)