import numpy as np

class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Fast FFT‑based convolution.

        Uses NumPy's FFT routines to compute the convolution of two real signals.
        Supports 'full', 'same', and 'valid' modes.

        :param problem: dict with keys:
                        - 'signal_x': list or array of real numbers
                        - 'signal_y': list or array of real numbers
                        - 'mode': str, one of 'full', 'same', 'valid' (default 'full')
        :return: dict with key 'convolution' containing the real result array.
        """
        x = np.asarray(problem['signal_x'], dtype=np.float64)
        y = np.asarray(problem['signal_y'], dtype=np.float64)
        mode = problem.get('mode', 'full').lower()
        nx, ny = len(x), len(y)

        # Choose length for FFT: next power of two for speed
        n_out = nx + ny - 1
        n_fft = 1 << (n_out - 1).bit_length()

        # Forward FFTs
        X = np.fft.rfft(x, n_fft)
        Y = np.fft.rfft(y, n_fft)

        # Element‑wise multiplication
        Z = X * Y

        # Inverse FFT and real part
        result = np.fft.irfft(Z, n_fft)[:n_out]

        # Slice according to mode
        if mode == 'same':
            start = (ny - 1) // 2
            result = result[start:start + nx]
        elif mode == 'valid':
            result = result[ny - 1: nx]
        # else 'full' returns whole result

        return {'convolution': result}