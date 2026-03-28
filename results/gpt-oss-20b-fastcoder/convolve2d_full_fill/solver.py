import numpy as np

class Solver:

    @staticmethod
    def _fft_convolve2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Convolve two 2D arrays using FFT.  Assumes all data are real
        and the result is also real.  Performs a full convolution
        equivalent to ``scipy.signal.convolve2d`` with mode='full'
        and boundary='fill' (i.e. zero padding).
        """
        # Determine output shape for full convolution
        out_shape = (a.shape[0] + b.shape[0] - 1,
                     a.shape[1] + b.shape[1] - 1)

        # Select next power‑of‑two size for efficient FFT
        fft_shape = [np.power(2, int(np.ceil(np.log2(dim)))) for dim in out_shape]

        # FFT of padded inputs
        Fa = np.fft.rfftn(a, fft_shape)
        Fb = np.fft.rfftn(b, fft_shape)

        # Element‑wise product in frequency domain
        Fc = Fa * Fb

        # Inverse FFT to get spatial domain result
        conv_full = np.fft.irfftn(Fc, fft_shape)

        # Truncate to the exact output size
        conv_full = conv_full[:out_shape[0], :out_shape[1]]

        return conv_full

    def solve(self, problem: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Compute the 2D convolution of arrays a and b using a fast FFT
        implementation.  The function mimics ``scipy.signal.convolve2d``'s
        ``mode='full'`` and ``boundary='fill'`` semantics.
        """
        a, b = problem
        # Use higher‑precision accumulator to reduce round‑off
        a = a.astype(np.float64, copy=False)
        b = b.astype(np.float64, copy=False)
        return self._fft_convolve2d(a, b)