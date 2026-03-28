import numpy as np
from scipy import signal
from typing import Tuple

class Solver:
    def __init__(self):
        # use full convolution
        self.mode = 'full'

    def _fast_fft_convolve(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Convolve two 1‑D real arrays using FFT. Works for large arrays."""
        n = a.shape[0] + b.shape[0] - 1
        # next power of two for efficient FFT
        N = 1 << (n - 1).bit_length()
        # FFTs
        A = np.fft.rfft(a, n=N)
        B = np.fft.rfft(b, n=N)
        # multiply and inverse
        y = np.fft.irfft(A * B, n=N)
        return y[:n]

    def solve(self, problem: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        a, b = problem
        # if both inputs are real and large, use FFT for speed,
        # otherwise fall back to numpy's convolution which is optimized
        # for small sizes and arbitrary dtypes.
        if a.dtype.kind == 'f' and b.dtype.kind == 'f' and a.size * b.size > 2**18:
            return self._fast_fft_convolve(a, b)
        return signal.convolve(a, b, mode=self.mode)