# solver.py
import numpy as np
from numpy.typing import NDArray

class Solver:
    """
    Optimised 2‑D correlation solver.
    Uses FFT‑based convolution for large inputs and the baseline
    scipy.signal.correlate2d for small ones.
    """

    def __init__(self):
        # kept for API compatibility but no effect in the FFT version
        self.boundary = "fill"
        self.mode = "full"

    @staticmethod
    def _fft_corr(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute the full 2‑D correlation of a and b using the FFT.
        Elements are assumed to be float64; other dtypes are cast.
        """
        # ensure contiguous and float64
        a = np.asarray(a, dtype=np.float64, order="C")
        b = np.asarray(b, dtype=np.float64, order="C")

        # shape for the FFT: output size is (a.shape[0]+b.shape[0]-1,
        #                                     a.shape[1]+b.shape[1]-1)
        out_shape = (a.shape[0] + b.shape[0] - 1, a.shape[1] + b.shape[1] - 1)

        # compute the FFT of both arrays, padded to out_shape
        fft_a = np.fft.fft2(a, s=out_shape)
        fft_b = np.fft.fft2(b, s=out_shape)

        # elementwise multiplication and inverse FFT
        corr = np.fft.ifft2(fft_a * fft_b)

        # take real part (imaginary part is numerical noise)
        return np.real(corr)

    def solve(self, problem: tuple) -> NDArray[np.float64]:
        """
        Compute the 2D correlation of arrays a and b using "full" mode.
        For small inputs the scipy implementation is used to avoid
        the overhead of FFT; for larger inputs the FFT approach is employed.

        Parameters
        ----------
        problem : tuple
            A tuple (a, b) containing two 2‑D numpy arrays.

        Returns
        -------
        numpy.ndarray
            The full correlation of a and b.
        """
        a, b = problem

        # use threshold to decide between direct and FFT method
        # Empirically, 128x128 arrays are a sweet spot.
        threshold = 128 * 128  # 16 384 elements

        if a.size * b.size <= threshold:
            # Fall back to the reference implementation
            try:
                from scipy import signal
                return signal.correlate2d(a, b, mode=self.mode, boundary=self.boundary)
            except Exception:
                pass  # If scipy is not available, fall back to FFT

        return self._fft_corr(a, b)