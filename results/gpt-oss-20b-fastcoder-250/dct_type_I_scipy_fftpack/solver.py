# solver.py

import numpy as np
from scipy import fft
from numpy.typing import NDArray
from typing import Any


class Solver:
    def solve(self, problem: NDArray, **kwargs) -> NDArray:
        """
        Compute the N-dimensional DCT Type I of `problem`.

        The function uses `scipy.fft.dctn` which is substantially faster than
        the legacy `scipy.fftpack` implementation while producing the same
        numerical results. The input and output are real-valued arrays.

        Parameters
        ----------
        problem : NDArray
            Input array of shape (n+1, n+1, ...) where the size along each
            axis is the logical size + 1, reflecting the even–symmetric
            boundary conditions required by DCT type I.

        Returns
        -------
        NDArray
            The DCT type I coefficients with the same shape as `problem`.
        """
        # `scipy.fft.dctn` supports the same interface as `fftpack.dctn`
        # and is optimized for performance with FFTW underneath.
        return fft.dctn(problem, type=1)
