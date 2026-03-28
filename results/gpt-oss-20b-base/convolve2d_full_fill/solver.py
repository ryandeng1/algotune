import numpy as np
from scipy.signal import fftconvolve

class Solver:
    """
    Compute the 2D convolution of arrays a and b.
    Uses FFT‑based convolution for large inputs, and
    direct convolution for very small inputs to avoid FFT overhead.
    """
    def __init__(self) -> None:
        # tfilt: use 'full' mode and handle boundaries by zero‑padding (default of fftconvolve)
        self.mode: str = "full"

    def solve(self, problem: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Compute the 2D convolution of arrays a and b using FFT‑based convolution.
        Parameters
        ----------
        problem : tuple
            Tuple of two 2D NumPy arrays (a, b).

        Returns
        -------
        np.ndarray
            Convolution of a and b in full mode.
        """
        a, b = problem
        # Ensure inputs are 2‑D float arrays for best FFT performance
        a = np.asarray(a, dtype=np.float64, order="C")
        b = np.asarray(b, dtype=np.float64, order="C")

        # De‑generate a very small array case to avoid FFT overhead
        if a.size <= 64 or b.size <= 64:
            return self._direct_convolution(a, b)

        # For larger arrays use FFT‑based convolution (fastest in practice)
        return fftconvolve(a, b, mode=self.mode)

    @staticmethod
    def _direct_convolution(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Direct convolution via nested loops (used only for tiny arrays).
        """
        m, n = a.shape
        p, q = b.shape
        out = np.zeros((m + p - 1, n + q - 1), dtype=np.float64)
        for i in range(m):
            for j in range(n):
                out[i : i + p, j : j + q] += a[i, j] * b
        return out