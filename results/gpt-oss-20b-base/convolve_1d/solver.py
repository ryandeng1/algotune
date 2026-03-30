import numpy as np
from numpy import fft

class Solver:
    """
    Fast convolution solver.
    For shorter vectors it delegates to :func:`numpy.convolve`.
    For longer vectors it uses FFT‐based convolution (which is usually faster).
    """

    SMALL_VEC_THRESHOLD = 128  # Empirical threshold for switching strategies

    def __init__(self) -> None:
        # Mode can be 'full', 'same', or 'valid'.
        self.mode = 'full'

    def _naive(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Direct convolution using numpy.convolve (C implementation)."""
        return np.convolve(a, b, mode=self.mode)

    def _fft_convolve(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """FFT based convolution for larger inputs."""
        # Determine the size of the padded array
        n = a.size + b.size - 1
        size = 1 << (n - 1).bit_length()  # next power of two

        # FFT of padded arrays
        A = fft.rfft(a, size)
        B = fft.rfft(b, size)

        # Element‑wise multiplication in frequency domain
        y = fft.irfft(A * B, size)

        # Truncate to 'full' length
        y = y[:n]

        # Adjust result according to requested mode
        if self.mode == 'full':
            return y
        elif self.mode == 'same':
            start = (n - max(a.size, b.size)) // 2
            return y[start:start + max(a.size, b.size)]
        else:  # valid
            start = a.size - 1
            end = b.size - 1
            return y[start:end]

    def solve(self, problem: tuple) -> np.ndarray:
        """
        Convolve two vectors.

        Parameters
        ----------
        problem : tuple
            A pair of 1‑D numpy arrays.

        Returns
        -------
        numpy.ndarray
            The convolution of the two input arrays.
        """
        a, b = problem
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)

        # Choose algorithm based on input size
        if a.size * b.size <= self.SMALL_VEC_THRESHOLD**2:
            return self._naive(a, b)
        return self._fft_convolve(a, b)