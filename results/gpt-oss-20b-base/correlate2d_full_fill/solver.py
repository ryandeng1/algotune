import numpy as np
from typing import Tuple

class Solver:

    def __init__(self):
        pass

    def solve(self, problem: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Compute the 2D correlation of arrays a and b using full mode (no boundary padding).
        This implementation uses FFT-based convolution for performance.
        """
        a, b = problem

        # Ensure inputs are float arrays
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)

        # Size of full correlation: shape_a + shape_b - 1
        shape_a = a.shape
        shape_b = b.shape
        out_shape = (shape_a[0] + shape_b[0] - 1,
                     shape_a[1] + shape_b[1] - 1)

        # Pad arrays to output size
        pad_a = np.zeros(out_shape, dtype=np.float64)
        pad_b = np.zeros(out_shape, dtype=np.float64)
        pad_a[:shape_a[0], :shape_a[1]] = a
        pad_b[:shape_b[0], :shape_b[1]] = b

        # FFTs
        fft_a = np.fft.fft2(pad_a)
        fft_b = np.fft.fft2(pad_b)

        # Correlation via pointwise multiplication with conjugate of b
        corr_fft = fft_a * np.conj(fft_b)

        # Inverse FFT to get correlation
        corr = np.fft.ifft2(corr_fft).real

        return corr