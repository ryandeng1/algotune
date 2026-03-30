# solver.py

import numpy as np
from numpy.typing import NDArray

# Use the newer scipy.fft module (available in recent SciPy versions).
# It delegates to an optimized backend such as FFTW and is typically
# faster than legacy scipy.fftpack.  Falling back to fftpack only if
# the new module is not available (for maximum compatibility).
try:
    from scipy.fft import dstn  # type: ignore[assignment]
except Exception:
    from scipy.fftpack import dstn  # type: ignore[assignment]

class Solver:
    """
    Solver for the N‑dimensional Discrete Sine Transform (DST) Type II.
    This implementation uses scipy's fast DFT implementation (`scipy.fft`)
    when available, otherwise falls back to `scipy.fftpack`.
    """

    def solve(self, problem: NDArray) -> NDArray:
        """
        Compute the N‑dimensional DST‑II of the input array.

        Parameters
        ----------
        problem : NDArray
            Input array of arbitrary dimensionality.

        Returns
        -------
        NDArray
            DST‑II result with the same shape as the input.
        """
        # Delegate to the optimized backend.  No additional overhead is
        # required beyond this single call.
        return dstn(problem, type=2)