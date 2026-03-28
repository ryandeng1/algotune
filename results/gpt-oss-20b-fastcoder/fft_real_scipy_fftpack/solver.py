from typing import Any
import numpy as np
from numpy.typing import NDArray

class Solver:
    """
    Fast N‑dimensional FFT implementation using :mod:`numpy.fft`.
    """
    def solve(self, problem: NDArray) -> NDArray:
        """
        Compute the N‑dimensional Fourier Transform of *problem*.

        Parameters
        ----------
        problem : NDArray
            Arbitrary‑dimensional array of real or complex numbers.

        Returns
        -------
        NDArray
            N‑dimensional FFT of *problem*.
        """
        # Use numpy's native FFT (FFTW backend when available) which
        # is typically faster than scipy.fftpack.fftn for large arrays.
        return np.fft.fftn(problem)