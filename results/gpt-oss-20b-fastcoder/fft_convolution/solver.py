import numpy as np

class Solver:
    def solve(self, problem):
        """
        Fast FFT‑based convolution without external dependencies.
        """
        x = np.asarray(problem['signal_x'], dtype=np.float64)
        y = np.asarray(problem['signal_y'], dtype=np.float64)
        mode = problem.get('mode', 'full')

        n = x.shape[0] + y.shape[0] - 1
        # Next power of two for efficient FFT
        fft_len = 1 << (n - 1).bit_length()

        # FFT of zero‑padded signals
        X = np.fft.rfft(x, n=fft_len)
        Y = np.fft.rfft(y, n=fft_len)

        # Point‑wise multiplication and inverse FFT
        conv = np.fft.irfft(X * Y, n=fft_len)

        if mode == 'full':
            result = conv[:n]
        elif mode == 'valid':
            result = conv[n - 1:]
        else:  # 'same'
            start = (n - 1) // 2
            result = conv[start:start + max(x.size, y.size)]

        return {"convolution": result.tolist()}