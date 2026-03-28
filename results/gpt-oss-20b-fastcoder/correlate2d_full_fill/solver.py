from typing import Tuple
import numpy as np


class Solver:
    """
    Optimized 2‑D correlation using FFT.  The implementation
    directly performs the convolution theorem:
        corr = ifft2( fft(a) * conj(fft(b)) )
    which is much faster than the naïve approach for large inputs.
    """

    def __init__(self):
        pass  # No runtime‑configurable state required

    @staticmethod
    def _fftcorr2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Return the full 2‑D correlation of a with b."""
        # Compute output size (full mode)
        a_shape = a.shape
        b_shape = b.shape
        out_shape = (a_shape[0] + b_shape[0] - 1,
                     a_shape[1] + b_shape[1] - 1)

        # FFT sizes (next pow of 2 for speed)
        fft_shape = tuple(1 << (n - 1).bit_length() for n in out_shape)

        A = np.fft.rfftn(a, fft_shape, norm=None)
        B = np.fft.rfftn(b, fft_shape, norm=None)
        # Correlation uses conj(B)
        corr = np.fft.irfftn(A * np.conj(B), fft_shape)
        # Trim to the exact output size
        return corr[:out_shape[0], :out_shape[1]]

    def solve(self, problem: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        a, b = problem
        return self._fftcorr2d(a, b)