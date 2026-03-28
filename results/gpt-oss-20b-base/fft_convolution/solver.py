import numpy as np
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List]:
        """
        Compute the linear convolution of two real sequences using FFT.
        Chosen for speed: only uses NumPy (no SciPy dependency) and
        handles both 'full' and 'same' modes.
        """
        # Convert inputs to 1‑D float arrays
        x = np.asarray(problem["signal_x"], dtype=np.float64, copy=False)
        y = np.asarray(problem["signal_y"], dtype=np.float64, copy=False)

        # Lengths of the inputs
        nx, ny = x.shape[0], y.shape[0]
        # Size of the FFT: next power of two of nx + ny - 1
        nfft = 1 << (nx + ny - 1).bit_length()

        # Forward FFT of zero‑padded inputs
        X = np.fft.rfft(x, nfft)
        Y = np.fft.rfft(y, nfft)

        # Element‑wise product in frequency domain and inverse FFT
        conv = np.fft.irfft(X * Y, nfft)[: nx + ny - 1]

        mode = problem.get("mode", "full").lower()
        if mode == "full":
            result = conv
        elif mode == "same":
            start = (ny - 1) // 2
            result = conv[start:start + nx]
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        # Return as a plain Python list
        return {"convolution": result.tolist()}