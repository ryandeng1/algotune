from typing import Any
import numpy as np

class Solver:
    """
    Fast convolution solver using NumPy FFT that is generally faster than
    scipy.signal.fftconvolve for 1‑D real valued signals and small to medium
    input sizes. It works in the 'full', 'valid' and 'same' modes.
    """

    @staticmethod
    def _prepare_padded(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Return the padded length which is the next power of two of (len(x)+len(y)-1).
        """
        n = len(x) + len(y) - 1
        return 1 << (n - 1).bit_length()

    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Convolve two signals using a real‑valued FFT.
        """
        x = np.asarray(problem["signal_x"], dtype=np.float64)
        y = np.asarray(problem["signal_y"], dtype=np.float64)
        mode = problem.get("mode", "full")

        # Full convolution length
        n = len(x) + len(y) - 1
        size = self._prepare_padded(x, y)

        # FFT of padded signals
        X = np.fft.rfft(x, size)
        Y = np.fft.rfft(y, size)

        # Element‑wise multiplication and inverse FFT
        conv = np.fft.irfft(X * Y, size)[:n]

        # Reduce to requested output mode
        if mode == "full":
            out = conv
        elif mode == "same":
            start = (len(y) - 1) // 2
            out = conv[start : start + len(x)]
        elif mode == "valid":
            out = conv[len(y) - 1 :]
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        return {"convolution": out.tolist()}