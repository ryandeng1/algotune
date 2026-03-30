# solver.py
from typing import Any
import numpy as np
import scipy.fft as fft
from numpy.typing import NDArray

class Solver:
    """
    Optimised N‑dimensional Discrete Cosine Transform (DCT) Type I solver.

    The implementation uses :func:`scipy.fft.dctn`, which is backed by the
    highly optimised DCT routines in the bundled FFTW/FFTW3 library and
    operates directly on the NumPy array without the intermediate
    ``scipy.fftpack`` wrapper.  It also accepts any array‑like input that
    can be coerced to ``numpy.ndarray``.
    """

    def solve(self, problem: NDArray) -> NDArray:
        """
        Compute the N‑dimensional DCT Type I of *problem*.

        Parameters
        ----------
        problem : NDArray
            Input array of arbitrary shape.

        Returns
        -------
        NDArray
            The DCT‑I transform of the input.
        """
        # Ensure the input is a NumPy array (avoids temp copies if already ndarray)
        arr = np.asarray(problem)
        return fft.dctn(arr, type=1, axis=None, norm=None)