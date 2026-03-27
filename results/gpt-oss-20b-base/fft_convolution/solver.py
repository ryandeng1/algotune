from typing import Any, Dict, List
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Compute the convolution of two 1-D signals using the Fast Fourier Transform.
        Implements the full, valid, and same modes that scipy.signal.fftconvolve
        supports.
        """
        x = np.asarray(problem["signal_x"], dtype=np.float64)
        y = np.asarray(problem["signal_y"], dtype=np.float64)

        mode = problem.get("mode", "full")

        # Length of the convolution
        n = x.size + y.size - 1

        # Pad to the next power of two for faster FFT on most hardware
        m = int(2 ** np.ceil(np.log2(n)))
        X = np.fft.rfft(x, m)
        Y = np.fft.rfft(y, m)
        conv = np.fft.irfft(X * Y, m)[:n]

        if mode == "full":
            result = conv
        elif mode == "valid":
            # Valid length is max(len(x), len(y)) - min(len(x), len(y)) + 1
            vlen = max(x.size, y.size) - min(x.size, y.size) + 1
            start = n - vlen
            result = conv[start:start + vlen]
        elif mode == "same":
            # Same length as the longer input, centered
            if x.size >= y.size:
                start = (y.size - 1) // 2
                result = conv[start:start + x.size]
            else:
                start = (x.size - 1) // 2
                result = conv[start:start + y.size]
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        return {"convolution": result.tolist()}