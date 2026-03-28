import numpy as np

class Solver:
    def __init__(self):
        # parameters are kept for API compatibility but are not used
        self.boundary = 'fill'
        self.mode = 'full'

    @staticmethod
    def _next_power_of_two(x: int) -> int:
        """Return the next power of two greater than or equal to x."""
        return 1 << (x - 1).bit_length()

    def solve(self, problem: tuple) -> np.ndarray:
        """
        Compute the 2D convolution of arrays a and b using FFT-based
        "full" mode. This implementation is typically faster than
        scipy.signal.convolve2d for large inputs.

        Parameters
        ----------
        problem : tuple
            A tuple containing two 2D NumPy arrays (a, b).

        Returns
        -------
        ndarray
            The full 2D convolution result.
        """
        a, b = problem
        if a is None or b is None:
            raise ValueError("Input arrays must not be None.")

        # Ensure inputs are float64 for numerical stability
        a = np.asarray(a, dtype=np.float64, order='C')
        b = np.asarray(b, dtype=np.float64, order='C')

        # Shape of the output for 'full' convolution
        out_shape = (a.shape[0] + b.shape[0] - 1, a.shape[1] + b.shape[1] - 1)

        # Compute FFT sizes (next power of two for speed)
        fft_shape = (
            self._next_power_of_two(out_shape[0]),
            self._next_power_of_two(out_shape[1]),
        )

        # Zero‑pad inputs to fft_shape
        A_fft = np.fft.rfft2(a, s=fft_shape)
        B_fft = np.fft.rfft2(b, s=fft_shape)

        # Element‑wise multiplication in frequency domain
        C_fft = A_fft * B_fft

        # Inverse FFT and crop to the full convolution size
        conv_full = np.fft.irfft2(C_fft, s=fft_shape)[: out_shape[0], : out_shape[1]]

        return conv_full