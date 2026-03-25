import numpy as np
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Compute convolution of two real signals using FFT."""
        # Extract inputs
        x = np.asarray(problem.get("signal_x", []), dtype=np.float64)
        y = np.asarray(problem.get("signal_y", []), dtype=np.float64)
        mode = problem.get("mode", "full")

        # Handle empty inputs
        if x.size == 0 or y.size == 0:
            return {"result": np.array([], dtype=np.float64)}

        # Determine output length
        if mode == "full":
            out_len = x.size + y.size - 1
        elif mode == "same":
            out_len = max(x.size, y.size)
        elif mode == "valid":
            out_len = max(0, min(x.size, y.size) - max(x.size, y.size) + 1)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        # Compute the next power of two for efficient FFT
        n = x.size + y.size - 1
        fft_size = 1 << (n - 1).bit_length()

        # FFT of zero-padded signals
        X = np.fft.rfft(x, n=fft_size)
        Y = np.fft.rfft(y, n=fft_size)

        # Element-wise multiplication in frequency domain
        Z = X * Y

        # Inverse FFT to get convolution result
        conv = np.fft.irfft(Z, n=fft_size)

        # Trim to the required output length
        conv = conv[:out_len]

        # For 'same' mode, center the output
        if mode == "same":
            start = (conv.size - out_len) // 2
            conv = conv[start:start + out_len]

        # For 'valid' mode, already correct length
        return {"result": conv}
