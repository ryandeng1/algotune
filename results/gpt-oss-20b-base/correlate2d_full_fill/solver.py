import numpy as np

class Solver:
    """
    Efficient 2‑D correlation using FFT.
    Supports "full" mode with zero padding (boundary = "fill").
    """
    def _next_fast_len(self, n):
        """Return the next power‑of‑two length for efficient FFT."""
        return 1 << (n - 1).bit_length()

    def solve(self, problem: tuple) -> np.ndarray:
        a, b = problem
        # Ensure float type for FFT
        a = np.asarray(a, dtype=np.float64, order="C")
        b = np.asarray(b, dtype=np.float64, order="C")

        # Sizes for full correlation
        out_shape = (a.shape[0] + b.shape[0] - 1,
                     a.shape[1] + b.shape[1] - 1)

        # Pad to next power‑of‑two for fast FFT (optional but often faster)
        fft_shape = (self._next_fast_len(out_shape[0]),
                     self._next_fast_len(out_shape[1]))

        # FFT of zero‑padded arrays
        fa = np.fft.rfftn(a, fft_shape)
        fb = np.fft.rfftn(b, fft_shape)

        # Element‑wise multiplication (correlation = flip b)
        # Flip b over both axes
        fb_flip = np.fft.ifftshift(fb)
        corr_fft = fa * fb_flip

        # Inverse FFT to get correlation
        corr = np.fft.irfftn(corr_fft, fft_shape)

        # Take real part and crop to full output size
        corr = np.real(corr[:out_shape[0], :out_shape[1]])

        return corr