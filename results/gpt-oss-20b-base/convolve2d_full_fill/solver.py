# solver.py
import numpy as np
from numpy.fft import rfftn, irfftn

class Solver:
    """
    Fast 2D full convolution using FFT.  Works for any real-valued inputs
    and matches the behavior of scipy.signal.convolve2d with mode='full'
    and boundary='fill'.
    """
    @staticmethod
    def _next_pow2(x: int) -> int:
        """Return the next power of two greater than or equal to x."""
        return 1 << (x - 1).bit_length()

    def solve(self, problem: tuple) -> np.ndarray:
        """Return full convolution of a and b using FFT."""
        a, b = problem
        # Dimensions
        ha, wa = a.shape
        hb, wb = b.shape
        # Full output size
        out_h = ha + hb - 1
        out_w = wa + wb - 1

        # Target size for FFT (next power of 2 for speed)
        fft_h = self._next_pow2(out_h)
        fft_w = self._next_pow2(out_w)

        # Pad inputs
        pad_a = np.zeros((fft_h, fft_w), dtype=a.dtype)
        pad_b = np.zeros((fft_h, fft_w), dtype=b.dtype)
        pad_a[:ha, :wa] = a
        pad_b[:hb, :wb] = b

        # Forward FFT (real to complex)
        fa = rfftn(pad_a, s=(fft_h, fft_w))
        fb = rfftn(pad_b, s=(fft_h, fft_w))

        # Element-wise multiplication in frequency domain
        fc = fa * fb

        # Inverse FFT
        conv_full = irfftn(fc, s=(fft_h, fft_w))

        # Crop to full output size
        result = conv_full[:out_h, :out_w]
        return result
