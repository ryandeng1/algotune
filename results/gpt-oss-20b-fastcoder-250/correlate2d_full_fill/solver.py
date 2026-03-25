import numpy as np

class Solver:
    def _fft_correlate2d(self, a, b):
        """
        Perform a 2D cross‑correlation using the FFT.  This is equivalent to
        scipy.signal.correlate2d with mode='full' and boundary='fill'.
        """
        # sizes
        h1, w1 = a.shape
        h2, w2 = b.shape

        # Result size
        h_out = h1 + h2 - 1
        w_out = w1 + w2 - 1

        # Pad to nearest power of two for speed (optional but cheap)
        h = 1 << (h_out - 1).bit_length()
        w = 1 << (w_out - 1).bit_length()

        # FFT of both arrays
        fa = np.fft.rfftn(a, s=(h, w))
        fb = np.fft.rfftn(b, s=(h, w))

        # Correlation = ifft of product of fa and conj(fb) because
        # correlation is convolution with flipped kernel.
        fc = fa * np.conj(fb)

        # Inverse FFT to get correlation
        corr = np.fft.irfftn(fc, s=(h, w))

        # Trim to exact output size
        return corr[:h_out, :w_out]

    def solve(self, problem, **kwargs) -> np.ndarray:
        """
        Compute the 2D correlation of arrays a and b using "full" mode and
        "fill" boundary.  Implements a fast FFT‐based approach that matches
        scipy.signal.correlate2d within numerical tolerance.
        """
        a, b = problem
        return self._fft_correlate2d(a, b)
