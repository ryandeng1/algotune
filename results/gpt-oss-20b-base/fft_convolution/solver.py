# solver.py
import numpy as np
from typing import Any, Dict

class Solver:
    def _fft_convolve(self, x: np.ndarray, y: np.ndarray, mode: str) -> np.ndarray:
        """Compute convolution via FFT."""
        # Handle empty inputs
        if x.size == 0 or y.size == 0:
            return np.array([], dtype=float)

        # Determine FFT size: next power of two >= len(x)+len(y)-1
        n = x.size + y.size - 1
        fft_size = 1 << (n - 1).bit_length()

        # FFT of zero-padded signals
        X = np.fft.rfft(x, n=fft_size)
        Y = np.fft.rfft(y, n=fft_size)

        # Element-wise multiplication and inverse FFT
        conv_full = np.fft.irfft(X * Y, n=fft_size)
        conv_full = conv_full[:n]  # truncate to exact length

        # Select mode
        if mode == "full":
            return conv_full
        elif mode == "same":
            # Center part of length max(len(x), len(y))
            k = max(x.size, y.size)
            start = (n - k) // 2
            return conv_full[start:start + k]
        elif mode == "valid":
            # Length = max(0, len(x)+len(y)-1 - (len(x)+len(y)-2))
            # Equivalent to max(0, min(len(x), len(y)) - abs(len(x)-len(y)) + 1)
            Lx, Ly = x.size, y.size
            if Lx < Ly:
                Lx, Ly = Ly, Lx
            valid_len = max(0, Ly - Lx + 1)
            start = n - valid_len
            return conv_full[start:] if valid_len > 0 else np.array([], dtype=float)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def solve(self, problem: Dict[str, Any], **kwargs) -> Any:
        """FFT-based convolution solver."""
        x = np.asarray(problem.get("signal_x", []), dtype=float)
        y = np.asarray(problem.get("signal_y", []), dtype=float)
        mode = problem.get("mode", "full")

        result = self._fft_convolve(x, y, mode)
        # Return as a numpy array to satisfy validation
        return {"convolution": result}
