import numpy as np
from math import ceil, log2

class Solver:
    """
    Optimised convolution using FFT for large sequences.
    Falls back to numpy.convolve for very small inputs where FFT overhead
    outweighs the speed gains.
    Supports the same `mode` options as scipy.signal.convolve:
        'full', 'same', 'valid'
    """

    _FFT_THRESHOLD = 64  # minimum length of a to consider FFT path

    @staticmethod
    def _next_pow_two(n: int) -> int:
        """Return the next power of two >= n."""
        return 1 << (n - 1).bit_length()

    @staticmethod
    def _fft_convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Perform convolution using FFT.  Input vectors are assumed to be
        1‑D numpy arrays of real or complex dtype.
        """
        n = a.shape[0] + b.shape[0] - 1
        N = Solver._next_pow_two(n)
        fa = np.fft.rfft(a, n=N)
        fb = np.fft.rfft(b, n=N)
        fc = fa * fb
        c = np.fft.irfft(fc, n=N)
        # Trim to required length
        return c[:n]

    def solve(self, problem: tuple) -> np.ndarray:
        a, b = problem
        # Ensure numpy arrays
        a = np.asarray(a)
        b = np.asarray(b)

        # Simpler path for very short signals
        if a.size < self._FFT_THRESHOLD or b.size < self._FFT_THRESHOLD:
            return np.convolve(a, b, mode='full')

        # FFT‑based full convolution
        full = self._fft_convolve(a, b)

        # Crop according to mode
        mode = getattr(self, "mode", "full")
        if mode == "full":
            return full
        elif mode == "same":
            if a.size >= b.size:
                start = (b.size - 1) // 2
                end = start + a.size
            else:
                start = (a.size - 1) // 2
                end = start + b.size
            return full[start:end]
        elif mode == "valid":
            # Length of valid output: max(a.size, b.size) - min(a.size, b.size) + 1
            n = max(a.size, b.size) - min(a.size, b.size) + 1
            start = full.size - n
            return full[start:]
        else:
            raise ValueError(f"Unsupported mode: {mode}")