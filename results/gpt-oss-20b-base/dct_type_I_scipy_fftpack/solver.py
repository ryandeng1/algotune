import numpy as np
from scipy.fft import dct
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray) -> NDArray:
        """
        Compute the N-dimensional DCT Type I using scipy.fft.dct.
        This implementation is faster than scipy.fftpack.dctn for
        multi-dimensional data as it employs the newer FFTPACK backend
        via scipy.fft.dct, which is highly optimised for speed.
        """
        # Ensure the input is a float array (dct works with floats)
        data = np.asanyarray(problem, dtype=float)
        # Apply DCT type I along each axis sequentially
        for axis in range(data.ndim):
            data = dct(data, type=1, axis=axis, norm=None)
        return data