# solver.py
from __future__ import annotations
from typing import Any, Dict, List
import numpy as np

class Solver:
    """
    Efficient convolution solver using real‑valued FFT.
    The implementation avoids the overhead of scipy.signal.fftconvolve
    by performing the FFT convolution directly with NumPy's fast FFT routines.
    """

    @staticmethod
    def _next_pow2(n: int) -> int:
        """Return the next power of two greater than or equal to n."""
        return 1 << (n - 1).bit_length()

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List]:
        """
        Convolve two 1‑D signals using a real‑valued FFT.

        Parameters
        ----------
        problem : dict
            Should contain:
            * ``signal_x``: sequence of samples to be convolved.
            * ``signal_y``: sequence of samples to be convolved.
            * ``mode``: one of ``'full'`` (default), ``'same'``, or ``'valid'``.

        Returns
        -------
        dict
            ``{"convolution": [result_as_list]}``
        """
        # Convert inputs to contiguous float64 arrays
        x = np.asarray(problem["signal_x"], dtype=np.float64, copy=False)
        y = np.asarray(problem["signal_y"], dtype=np.float64, copy=False)
        mode = problem.get("mode", "full").lower()

        nx, ny = x.shape[0], y.shape[0]
        n = nx + ny - 1  # length of full convolution

        # Zero‑pad to the next power of two for FFT efficiency
        N = self._next_pow2(n)
        X = np.fft.rfft(x, n=N)
        Y = np.fft.rfft(y, n=N)
        Z = X * Y
        conv_full = np.fft.irfft(Z, n=N)[:n]  # truncate to actual length

        if mode == "full":
            result = conv_full
        elif mode == "same":
            start = (ny - 1) // 2
            result = conv_full[start:start + nx]
        elif mode == "valid":
            result = conv_full[ny - 1 : nx]
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        return {"convolution": result.tolist()}