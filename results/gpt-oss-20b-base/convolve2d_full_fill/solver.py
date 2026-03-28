import numpy as np

class Solver:
    def solve(self, problem: tuple) -> np.ndarray:
        """
        Compute the 2D convolution of arrays a and b using the FFT-based
        algorithm. This is typically faster than direct convolution for large
        arrays.

        :param problem: A tuple (a, b) of 2D numpy arrays.
        :return: A 2D array containing the full convolution result.
        """
        a, b = problem
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("Inputs must be 2D arrays")

        # Determine shape of the full convolution
        m, n = a.shape
        p, q = b.shape
        out_shape = (m + p - 1, n + q - 1)

        # Pad arrays to the output size
        fft_shape = [2 ** int(np.ceil(np.log2(s))) for s in out_shape]
        A = np.fft.rfft2(a, s=fft_shape)
        B = np.fft.rfft2(b, s=fft_shape)

        # Element-wise multiplication in frequency domain
        C = A * B

        # Inverse transform to spatial domain
        conv = np.fft.irfft2(C, s=fft_shape)

        # Trim to the exact size of the full convolution
        conv = conv[:out_shape[0], :out_shape[1]]

        return conv.astype(np.float64)