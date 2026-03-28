import numpy as np

class Solver:
    def __init__(self):
        # only at init we can cache nothing
        self.mode = 'full'

    @staticmethod
    def _fft_convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Convolution via FFT. Works in all modes.
        """
        # determine the size needed for full convolution
        n = a.shape[0] + b.shape[0] - 1
        # next power of two for efficient FFT
        fft_size = 1 << (n - 1).bit_length()

        # FFT of zero‑padded inputs
        A = np.fft.rfft(a, fft_size)
        B = np.fft.rfft(b, fft_size)

        # element‑wise multiplication
        C = A * B

        # inverse FFT and discard imaginary part
        conv = np.fft.irfft(C, fft_size)

        # crop to the correct mode
        if Solver._mode_flag == 'full':
            return conv[:n]
        elif Solver._mode_flag == 'valid':
            # valid sized output
            start = a.size - 1
            end = start + a.size - b.size + 1
            return conv[start:end]
        elif Solver._mode_flag == 'same':
            # same sized output as a
            start = (n - a.size) // 2
            return conv[start:start + a.size]
        else:
            raise ValueError(f"Unknown mode: {Solver._mode_flag}")

    def solve(self, problem: tuple) -> np.ndarray:
        """
        Optimised convolution of 1‑D sequences `a` and `b`.
        Uses FFT based convolution for large inputs, otherwise defaults
        to `numpy`'s efficient implementation.
        """
        a, b = problem

        # ensure numpy arrays
        a = np.asarray(a, dtype=float).ravel()
        b = np.asarray(b, dtype=float).ravel()

        # small inputs: fall back to numpy's convolution
        if a.size * b.size <= 5000:
            return np.convolve(a, b, mode=self.mode)

        # set mode flag for static helper
        Solver._mode_flag = self.mode
        return self._fft_convolve(a, b)