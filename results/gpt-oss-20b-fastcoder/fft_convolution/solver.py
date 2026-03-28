import numpy as np
from typing import Any, Dict, List

class Solver:
    # next power of two for efficient FFT
    @staticmethod
    def _next_pow2(n: int) -> int:
        return 1 << (n - 1).bit_length()

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List]:
        x = np.asarray(problem["signal_x"], dtype=np.float64)
        y = np.asarray(problem["signal_y"], dtype=np.float64)
        mode = problem.get("mode", "full")

        if mode == "full":
            n = x.size + y.size - 1
            size = self._next_pow2(n)
            fx = np.fft.rfft(x, size)
            fy = np.fft.rfft(y, size)
            conv = np.fft.irfft(fx * fy, size)[:n]
        elif mode == "same":
            # same length as x
            n = x.size
            size = self._next_pow2(n + y.size - 1)
            fx = np.fft.rfft(x, size)
            fy = np.fft.rfft(y, size)
            conv_full = np.fft.irfft(fx * fy, size)
            start = (y.size - 1) // 2
            conv = conv_full[start:start + n]
        elif mode == "valid":
            n = x.size - y.size + 1
            if n < 1:
                conv = np.array([], dtype=np.float64)
            else:
                size = self._next_pow2(x.size + y.size - 1)
                fx = np.fft.rfft(x, size)
                fy = np.fft.rfft(y, size)
                conv_full = np.fft.irfft(fx * fy, size)
                start = y.size - 1
                conv = conv_full[start:start + n]
        else:
            raise ValueError(f"Unsupported mode '{mode}'")

        return {"convolution": conv.tolist()}