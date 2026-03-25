import numpy as np
from scipy.fft import dctn
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray, **kwargs) -> NDArray:
        """
        Compute the N-dimensional DCT Type I using SciPy's FFT module.
        The SciPy FFT implementation is faster than fftpack for large data.
        """
        # Using the newer scipy.fft here which wraps FFTW and generally has
        # better performance than scipy.fftpack. The API for dctn is identical.
        return dctn(problem, type=1)
