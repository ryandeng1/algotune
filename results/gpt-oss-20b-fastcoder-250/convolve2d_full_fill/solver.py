import numpy as np
from typing import Tuple

class Solver:
    def __init__(self) -> None:
        # Convolution mode and boundary are fixed for this task
        self.mode = "full"
        self.boundary = "fill"

    def _fft2convolve(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Perform a 2‑D convolution using the FFT.

        Parameters
        ----------
        a, b : np.ndarray
            2–D input arrays.

        Returns
        -------
        np.ndarray
            The full convolution of `a` and `b`.
        """
        # Pad both arrays to the same shape: result size is (len(a)+len(b)-1, ...)
        out_shape = (a.shape[0] + b.shape[0] - 1,
                     a.shape[1] + b.shape[1] - 1)

        # Forward FFT of both arrays
        fa = np.fft.rfftn(a, out_shape)
        fb = np.fft.rfftn(b, out_shape)

        # Element‑wise product in frequency domain
        fc = fa * fb

        # Inverse FFT to obtain the convolution
        result = np.fft.irfftn(fc, out_shape)

        # Convolution of real inputs results in real output; cast to float32 for speed
        return result.astype(np.float32, copy=False)

    def solve(self, problem: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Compute the 2D convolution of arrays `a` and `b` using the FFT.

        Parameters
        ----------
        problem : tuple
            A tuple ``(a, b)`` of 2‑D float or complex arrays.

        Returns
        -------
        np.ndarray
            The convolution result in "full" mode.
        """
        a, b = problem
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("Inputs must be 2‑D arrays")

        # SciPy's signal.convolve2d has overhead; use pure NumPy FFT instead
        return self._fft2convolve(a, b)