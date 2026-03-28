import numpy as np

class Solver:
    """Fast 2‑D convolution in full mode using the FFT algorithm."""

    def __init__(self):
        # Parameters are preserved only for API compatibility
        self.boundary = "fill"
        self.mode = "full"

    def solve(self, problem: tuple) -> np.ndarray:
        """
        Compute the 2D convolution of arrays `a` and `b` in full mode.
        Uses the FFT based implementation from NumPy for speed.
        """
        a, b = problem

        # Ensure the input is in float for the FFT and that the shapes
        # are at least 1‑dimensional
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)

        # Size of the output: (M+K-1, N+L-1)
        out_shape = (a.shape[0] + b.shape[0] - 1, a.shape[1] + b.shape[1] - 1)

        # Find power‑of‑two shape for efficient FFT
        fshape = [np.power(2, np.ceil(np.log2(sz)).astype(int)) for sz in out_shape]

        # Perform the FFT of the zero‑padded inputs
        fa = np.fft.rfftn(a, fshape)
        fb = np.fft.rfftn(b, fshape)

        # Element‑wise multiplication in frequency domain
        conv = np.fft.irfftn(fa * fb, fshape)

        # Slice to the actual output size
        conv = conv[:out_shape[0], :out_shape[1]]

        return conv