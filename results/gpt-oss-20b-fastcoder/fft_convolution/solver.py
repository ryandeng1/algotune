import numpy as np
from typing import Any, Dict, List

class Solver:
    @staticmethod
    def _fft_convolve(x: np.ndarray, y: np.ndarray, mode: str = 'full') -> np.ndarray:
        """Convolve two 1-D arrays using FFT with efficient padding."""
        nx, ny = x.shape[0], y.shape[0]
        n = nx + ny - 1
        # Power‑of‑two size for faster FFT
        m = 1 << (n - 1).bit_length()
        # FFT of both inputs
        fx = np.fft.rfftn(x, n=m)
        fy = np.fft.rfftn(y, n=m)
        # Element‑wise multiplication in the frequency domain
        fxy = fx * fy
        # Inverse FFT to obtain the linear convolution
        conv = np.fft.irfftn(fxy, n=m)
        # Trim to requested mode
        if mode == 'full':
            return conv[:n]
        elif mode == 'same':
            start = (ny - 1) // 2
            end = start + nx
            return conv[start:end]
        elif mode == 'valid':
            if nx < ny:
                raise ValueError("In 'valid' mode, len(x) must be >= len(y)")
            return conv[ny-1:nx]
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """Compute convolution of 'signal_x' and 'signal_y' using a fast FFT implementation."""
        x = np.asarray(problem['signal_x'], dtype=np.float64)
        y = np.asarray(problem['signal_y'], dtype=np.float64)
        mode = problem.get('mode', 'full')
        conv = self._fft_convolve(x, y, mode=mode)
        return {'convolution': conv.tolist()}