from typing import Tuple
import numpy as np

class Solver:
    """
    High‑performance 1‑D convolution dispatcher.

    The function picks the fastest convolution algorithm according
    to the input lengths:
      * For very small input (< 64) a direct algorithm is used.
      * For medium size (64 ≤ n < 8192) the classic NumPy
        convolution (which in turn uses a highly optimised
        C implementation) is applied.
      * For large inputs the FFT‑based method is employed
        which is asymptotically O(n log n).
    """

    __slots__ = ()  # no instance attributes needed

    def __init__(self, mode: str = "full"):
        """
        Parameters
        ----------
        mode : {'full', 'valid', 'same'}
            Convolution mode to match NumPy/Scipy semantics.
        """
        if mode not in {"full", "same", "valid"}:
            raise ValueError("mode must be one of 'full', 'same', 'valid'")
        self.mode = mode

    # -----------------------------------------------------------------------
    @staticmethod
    def _fft_convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Convolution via Welch‑overlap‑add using FFT."""
        n = a.size
        m = b.size
        size = n + m - 1
        # Pad to next power of two for speed
        fftlen = 1 << (size - 1).bit_length()
        # Compute FFTs
        A = np.fft.rfft(a, fftlen, dtype=np.complex128)
        B = np.fft.rfft(b, fftlen, dtype=np.complex128)
        # Point‑wise multiplication and inverse transform
        conv = np.fft.irfft(A * B, fftlen, dtype=np.complex128)
        return conv[:size].real

    # -----------------------------------------------------------------------
    def _apply_mode(self, conv: np.ndarray, m: int, n: int) -> np.ndarray:
        """Trim or pad the full convolution result to the requested mode."""
        if self.mode == "full":
            return conv
        elif self.mode == "same":
            start = (m - 1) // 2
            return conv[start:start + n]
        else:  # valid
            start = m - 1
            end = start + min(n, m) - 1
            return conv[start:end]

    # -----------------------------------------------------------------------
    def solve(self, problem: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        a, b = problem
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)

        if a.size == 0 or b.size == 0:
            # Edge case – mimic numpy/scipy behaviour
            if self.mode == "full":
                return np.array([], dtype=float)
            elif self.mode == "same":
                return np.zeros(a.size if a.size else b.size, dtype=float)
            else:  # valid
                return np.array([], dtype=float)

        # Decide on algorithm
        n, m = a.size, b.size
        if n * m <= 4096:  # small, use direct
            conv = np.dot(a[:, None], b[None, :]).reshape(n + m - 1)
        elif n * m <= 131072:  # medium, use np.convolve
            conv = np.convolve(a, b, mode="full")
        else:  # large, use FFT
            conv = self._fft_convolve(a, b)

        return self._apply_mode(conv, m, n)