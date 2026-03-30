# solver.py
from __future__ import annotations
from typing import Any
import numpy as np
from scipy.fft import dstn
from numpy.typing import NDArray

class Solver:
    """
    Optimised N‑dimensional Discrete Sine Transform (DST) implementation.
    Uses the high‑performance `scipy.fft.dstn` routine instead of the older
    `scipy.fftpack` version, and ensures the input is contiguous for
    maximal speed.
    """

    # Pre‑load the dstn function once so that each call doesn’t incur the
    # attribute lookup cost.
    _dstn = staticmethod(dstn)

    def __call__(self, problem: NDArray) -> NDArray:
        """
        Compute the N-dimensional DST Type II.
        Parameters
        ----------
        problem : NDArray
            Input array. The function does not modify this array.
        Returns
        -------
        NDArray
            The result of the DST computation; type is `float64`
            regardless of input dtype.
        """
        # Ensure the input is contiguous to avoid performance penalties
        # in the underlying C routine.
        arr = np.ascontiguousarray(problem, dtype=np.float64)
        # Call the pre‑loaded function directly for speed.
        return self._dstn(arr, type=2)

    # The original interface used a named method `solve`. We keep it for
    # compatibility while delegating to the faster ``__call__`` implementation.
    def solve(self, problem: NDArray) -> NDArray:
        return self.__call__(problem)