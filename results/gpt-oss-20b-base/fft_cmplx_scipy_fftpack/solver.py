# solver.py
from typing import Any
import numpy as np
from numpy.typing import NDArray

class Solver:
    """
    A light‑weight wrapper that computes the N‑dimensional Fast Fourier Transform
    using NumPy's highly optimised FFT routines. NumPy's implementation is
    typically faster than scipy.fftpack and is safe to use for all array
    types supported by NumPy.
    """

    def solve(self, problem: NDArray) -> NDArray:
        """
        Compute the N-dimensional FFT of *problem*.

        Parameters
        ----------
        problem : NDArray
            Input array. Can be real or complex, arbitrary shape and
            dimensionality.

        Returns
        -------
        NDArray
            The transformed array of the same shape as *problem*.
        """
        return np.fft.fftn(problem)