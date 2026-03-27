from typing import Any
import numpy as np


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Compute the convolution of two real sequences using the FFT method.
        This implementation avoids the overhead of scipy and works directly with NumPy.
        """
        # Extract signals and mode
        x = np.asarray(problem["signal_x"], dtype=float)
        y = np.asarray(problem["signal_y"], dtype=float)
        mode = problem.get("mode", "full")

        # Determine the length of the result
        if mode == "full":
            conv_len = len(x) + len(y) - 1
        elif mode == "valid":
            conv_len = max(len(x), len(y)) - min(len(x), len(y)) + 1
        elif mode == "same":
            conv_len = max(len(x), len(y))
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        # Zero‑pad signals to the next power of two for FFT efficiency
        n = 1 << (len(x) + len(y) - 1).bit_length()
        X = np.fft.rfft(x, n)
        Y = np.fft.rfft(y, n)
        conv_full = np.fft.irfft(X * Y, n)

        # Slice according to the requested mode
        if mode == "full":
            result = conv_full[:conv_len]
        elif mode == "same":
            start = (len(x) - conv_len) // 2
            result = conv_full[start : start + conv_len]
        else:  # valid
            start = len(x) - 1
            result = conv_full[start : start + conv_len]

        return {"convolution": result.tolist()}