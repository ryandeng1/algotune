import numpy as np
from numpy.fft import fft2, ifft2

class Solver:
    """
    Optimized 2‑D convolution using FFT.
    """
    def __init__(self):
        # parameters are kept for API compatibility but not used
        self.boundary = 'fill'
        self.mode = 'full'

    @staticmethod
    def _fft_convolve2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute 2‑D convolution via FFT. Works for real inputs.
        """
        # Handle trivial case
        if a.size == 0 or b.size == 0:
            return np.array([], dtype=a.dtype)

        # Determine output shape for full convolution
        out_shape = (a.shape[0] + b.shape[0] - 1,
                     a.shape[1] + b.shape[1] - 1)

        # Pad both arrays to the output shape
        fft_shape = [np.next_fast_len(s) for s in out_shape]
        A = np.zeros(fft_shape, dtype=np.complex128)
        B = np.zeros(fft_shape, dtype=np.complex128)
        A[:a.shape[0], :a.shape[1]] = a
        B[:b.shape[0], :b.shape[1]] = b

        # FFT, multiply, iFFT
        C = ifft2(fft2(A) * fft2(B)).real
        return C[:out_shape[0], :out_shape[1]]

    def solve(self, problem: tuple) -> np.ndarray:
        a, b = problem
        return self._fft_convolve2d(a, b)