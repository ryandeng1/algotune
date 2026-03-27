import numpy as np

class Solver:
    """
    Compute the 2‑D correlation of two arrays using the FFT for maximum performance.
    Works for the `"full"` mode and `"fill"` boundary which is equivalent to zero‑padding.
    """
    def __init__(self, mode: str = "full", boundary: str = "fill"):
        if mode != "full":
            raise NotImplementedError("Only 'full' mode is supported")
        if boundary != "fill":
            raise NotImplementedError("Only 'fill' boundary is supported")
        self.mode = mode
        self.boundary = boundary

    def solve(self, problem: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        a, b = problem
        # Cast to float for FFT precision
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)

        # Determine output shape for full correlation
        out_shape = (a.shape[0] + b.shape[0] - 1, a.shape[1] + b.shape[1] - 1)

        # Pad arrays to the output shape (FFT length)
        fft_shape = [np.max([size, out]) for size, out in zip((a.shape[0], a.shape[1]), out_shape)]
        N, M = fft_shape

        # FFT of padded arrays
        fa = np.fft.rfft2(a, s=(N, M))
        fb = np.fft.rfft2(b, s=(N, M))

        # Element‑wise multiplication for cross‑correlation (conjugate on second factor)
        fc = fa * np.conj(fb)

        # Inverse FFT to get correlation, then crop to full mode size
        corr = np.fft.irfft2(fc, s=(N, M))
        start0, start1 = b.shape[0] - 1, b.shape[1] - 1
        return corr[start0:start0 + out_shape[0], start1:start1 + out_shape[1]]