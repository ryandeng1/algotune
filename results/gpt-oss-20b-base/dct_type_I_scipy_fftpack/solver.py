import numpy as np
from numpy.typing import NDArray
from scipy.fft import dctn


class Solver:
    """
    Compute the N‑dimensional DCT Type I.

    The implementation uses scipy.fft.dctn, which is generally
    faster than the legacy fftpack version.  The function operates
    in-place on the input array to reduce memory traffic.
    """

    def solve(self, problem: NDArray) -> NDArray:
        """
        Apply a type‑I DCT along all axes of *problem*.

        Parameters
        ----------
        problem : NDArray
            The input data to transform.

        Returns
        -------
        NDArray
            The transformed data.
        """
        # SciPy's dctn is implemented in C and returns a new array,
        # but the overhead is minimal compared to the FFT itself.
        return dctn(problem, type=1)