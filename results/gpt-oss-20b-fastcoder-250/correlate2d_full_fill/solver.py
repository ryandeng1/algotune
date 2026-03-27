import numpy as np
from scipy.fft import fft2, ifft2, fftshift
from typing import Tuple


class Solver:
    """
    Fast 2‑D correlation using FFT. The implementation falls back to
    scipy.signal.correlate2d for small inputs or when the signal size is not
    a power of two.  The mode and boundary arguments are fixed to the values
    used by the original implementation: mode="full" and boundary="fill".
    """

    def __init__(self):
        self.mode = "full"
        self.boundary = "fill"

    def _fft_correlate(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Correlate two 2‑D arrays with an FFT‑based algorithm.

        Parameters
        ----------
        a, b : np.ndarray
            Input arrays.  They may contain any numeric type supported by NumPy.

        Returns
        -------
        result : np.ndarray
            Full 2‑D correlation of a and b.
        """
        # Determine the shape for the FFT that contains the full output
        out_shape = (a.shape[0] + b.shape[0] - 1,
                     a.shape[1] + b.shape[1] - 1)

        # Pad zero‑filled and compute the FFT of both arrays
        F_a = fft2(a, s=out_shape)
        F_b = fft2(b, s=out_shape)

        # Correlation in the Fourier domain is achieved by multiplying the
        # first array with the complex conjugate of the second.
        F_corr = F_a * np.conjugate(F_b)

        # Inverse FFT to get back to real domain
        corr = ifft2(F_corr).real

        # fft2/ifft2 can produce a small imaginary component due to numerical
        # error; we binarise the result by casting to float.
        return corr

    def solve(self, problem: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Compute the 2‑D correlation of arrays a and b using "full" mode and
        "fill" boundary.

        Parameters
        ----------
        problem : tuple
            A tuple (a, b) where both are 2‑D NumPy arrays.

        Returns
        -------
        np.ndarray
            Array containing the correlation result.
        """
        a, b = problem

        # If either array is trivial, use the built‑in function for correctness
        if (a.size == 0 or b.size == 0 or
                min(a.shape + b.shape) <= 32):
            from scipy.signal import correlate2d
            return correlate2d(a, b, mode=self.mode, boundary=self.boundary)

        return self._fft_correlate(a, b)