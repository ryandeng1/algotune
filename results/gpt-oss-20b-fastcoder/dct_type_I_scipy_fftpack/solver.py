# solver.py
import numpy as np
from scipy.fft import dctn
from numpy.typing import NDArray

class Solver:
    """
    Fast N‑dimensional Discrete Cosine Transform (DCT) Type I.

    Notes
    -----
    The implementation uses SciPy's newer FFT interface which is typically
    faster than :mod:`scipy.fftpack`.  The transform is computed in place
    when possible to avoid unnecessary memory allocations.
    """
    def __init__(self) -> None:
        # Cache the function for marginal speedup
        self._dctn = dctn

    def solve(self, problem: NDArray) -> NDArray:
        """
        Compute the N‑dimensional DCT Type I of *problem*.

        Parameters
        ----------
        problem : NDArray
            The input array to transform.

        Returns
        -------
        NDArray
            The DCT Type I of the input.
        """
        # SciPy's implementation accepts the array directly and returns the
        # transformed data.  We keep the data type of the input.
        return self._dctn(problem, type=1, norm=None, overwrite_x=True)