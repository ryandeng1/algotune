# solver.py
import numpy as np
from typing import Tuple

class Solver:
    """
    Efficient 2D convolution for large arrays using FFT. 
    The implementation mimics scipy.signal.convolve2d with boundary='fill' and mode='full'.
    """

    def __init__(self):
        # Utilise the default boundary and mode from scipy:
        self.boundary = 'fill'  # zero padding
        self.mode = 'full'      # convolution that returns the full overlap

    @staticmethod
    def _fft_convolve2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Perform 2D convolution via the FFT.

        Parameters
        ----------
        a : np.ndarray
            First input array.
        b : np.ndarray
            Second input array.

        Returns
        -------
        np.ndarray
            Convolution result.
        """
        # Ensure the inputs are 2-D
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("Both a and b must be 2-D arrays.")

        # Determine the shape of the output for full convolution
        out_shape = (a.shape[0] + b.shape[0] - 1, a.shape[1] + b.shape[1] - 1)

        # Pad each array to the output shape with zeros
        # Use np.fft.fft2 which requires power‑of‑two sized arrays for best performance
        # but will still work otherwise
        fft_shape = (
            1 << (out_shape[0] - 1).bit_length(),  # next power of two >= out_shape[0]
            1 << (out_shape[1] - 1).bit_length(),  # next power of two >= out_shape[1]
        )

        # FFT of zero‑padded arrays
        fa = np.fft.rfft2(a, fft_shape)
        fb = np.fft.rfft2(b, fft_shape)

        # Element‑wise multiplication in frequency domain
        fc = fa * fb

        # Inverse FFT to get the spatial domain convolution
        convolved = np.fft.irfft2(fc, fft_shape)

        # Slice to the exact output shape
        return convolved[:out_shape[0], :out_shape[1]]

    def solve(self, problem: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Compute the 2D convolution of arrays a and b using "full" mode and "fill" boundary.

        Parameters
        ----------
        problem : tuple
            A tuple (a, b) of 2-D numpy arrays.

        Returns
        -------
        np.ndarray
            The convolution result as a 2-D array of dtype float64.
        """
        a, b = problem
        # Use the FFT based convolution for speed.
        return self._fft_convolve2d(a, b)