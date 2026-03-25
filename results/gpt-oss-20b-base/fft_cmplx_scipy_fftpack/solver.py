# solver.py

import numpy as np
from numpy.typing import NDArray
from typing import Any

class Solver:
    """
    Fast implementation of an N-dimensional FFT for complex-valued matrices.
    The method simply delegates to numpy's highly optimized fftn routine, which
    internally uses FFTW on most platforms, providing superior speed over
    scipy's legacy fftpack implementation.
    """
    def solve(self, problem: NDArray, **kwargs) -> NDArray:
        """
        Compute the N-dimensional FFT of a complex-valued array.

        Parameters
        ----------
        problem : NDArray
            Input array of arbitrary dimensions containing complex numbers.

        Returns
        -------
        NDArray
            The N-dimensional FFT of the input array.
        """
        # Directly use numpy's fftn for maximum performance
        return np.fft.fftn(problem)
