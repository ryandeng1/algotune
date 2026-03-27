import numpy as np
from typing import Tuple

class Solver:
    def solve(self, problem: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Compute the 2D convolution of arrays a and b using "full" mode with a boundary
        equal to 0 ("fill") by using FFT-based convolution for better performance on
        large arrays.

        Parameters
        ----------
        problem : tuple
            A tuple (a, b) containing two 2D numpy arrays.

        Returns
        -------
        np.ndarray
            The convolution result in "full" mode.
        """
        a, b = problem

        # Ensure inputs are 2‑D arrays
        a = np.asarray(a)
        b = np.asarray(b)
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("Both inputs must be 2‑D matrices.")

        # Size of the output for full convolution
        out_shape = (a.shape[0] + b.shape[0] - 1, a.shape[1] + b.shape[1] - 1)

        # Determine FFT size (next power of two for speed / zero‑pad to out_shape)
        fft_shape = tuple(2**int(np.ceil(np.log2(os))) for os in out_shape)

        # Forward FFT of both arrays (zero‑padded to fft_shape)
        fa = np.fft.rfftn(a, fft_shape, axes=(0, 1))
        fb = np.fft.rfftn(b, fft_shape, axes=(0, 1))

        # Element‑wise multiplication in frequency domain
        prod = fa * fb

        # Inverse FFT to get convolution result
        conv = np.fft.irfftn(prod, fft_shape, axes=(0, 1))

        # Slice to the exact full convolution shape
        result = conv[:out_shape[0], :out_shape[1]]

        # Because of numerical errors the result might have very small
        # imaginary parts or be slightly off; we only return the real part
        # and cast to the same dtype as the input (preferably float128 for precision).
        return result.real.astype(np.result_type(a, b))