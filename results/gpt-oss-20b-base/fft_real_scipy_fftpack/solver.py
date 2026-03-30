# solver.py

"""
Optimized FFT solver.

This implementation uses :func:`numpy.fft.fftn`, which is
implemented in C and, for most use‑cases, is orders of
magnitude faster than the deprecated `scipy.fftpack.fftn`.
The code deliberately avoids any extra Python overhead,
keeping the public API identical to the original version.
"""

from __future__ import annotations

from typing import Any
import numpy as np
from numpy.typing import NDArray


class Solver:
    """
    Compute the N‑dimensional Fast Fourier Transform of an array.
    """

    @staticmethod
    def solve(problem: NDArray) -> NDArray:
        """
        Return the N‑dimensional FFT of ``problem``.

        Parameters
        ----------
        problem : NDArray
            Input array of any shape and complex or real dtype.

        Returns
        -------
        NDArray
            FFT of ``problem``.
        """
        # `np.fft.fftn` is a thin wrapper around a highly optimised
        # C implementation (via the alias `fftw` when available).
        # We simply delegate to it; no further conversion or copying
        # is necessary.
        return np.fft.fftn(problem)