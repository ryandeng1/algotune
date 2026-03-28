import numpy as np
from scipy import signal

class Solver:
    """Fast 2‑D convolution (full mode, fill boundary) using a hybrid FFT /
    direct algorithm for optimal performance."""
    __slots__ = ("boundary", "mode")

    def __init__(self):
        self.boundary = "fill"
        self.mode = "full"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def solve(self, problem: tuple) -> np.ndarray:
        """
        Compute the 2‑D convolution of arrays a and b using full mode and
        fill boundary. For large inputs the algorithm switches to an FFT
        based approach; otherwise it falls back to scipy.signal.convolve2d.
        """
        a, b = problem
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("Both arguments must be 2‑D arrays")

        # Threshold is empirically chosen. For small arrays direct method wins.
        if max(a.size, b.size) < 1_000_00:
            return signal.convolve2d(a, b, mode=self.mode, boundary=self.boundary)

        return self._fft_convolve2d(a, b)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _fft_convolve2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """FFT based full convolution."""
        # Pad to next power‑of‑two to avoid convolution wrap‑round issues
        shape = np.array(a.shape[::-1]) + np.array(b.shape[::-1]) - 1
        fft_shape = 2 ** np.ceil(np.log2(shape)).astype(int)

        fa = np.fft.rfftn(a, fft_shape, axes=(0, 1))
        fb = np.fft.rfftn(b, fft_shape, axes=(0, 1))
        f = fa * fb
        return np.fft.irfftn(f, shape, axes=(0, 1))