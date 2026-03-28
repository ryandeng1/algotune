import numpy as np
from typing import List, Tuple

class Solver:
    """
    Solve the upfirdn operation without relying on SciPy.

    For each tuple (h, x, up, down) the result is:

        y = downsample( convolve( upsample(x, up), h ) )

    where upsample inserts (up-1) zeros between consecutive samples of `x`
    and downsample keeps every `down`‑th sample of the filtered signal.
    """
    @staticmethod
    def _upsample(x: np.ndarray, up: int) -> np.ndarray:
        """Insert (up-1) zeros between elements of x."""
        if up == 1:
            return x
        n = x.size
        out = np.empty(n * up, dtype=x.dtype)
        out[::up] = x
        return out

    @staticmethod
    def _downsample(x: np.ndarray, down: int) -> np.ndarray:
        """Keep every down‑th sample of x."""
        if down == 1:
            return x
        return x[::down]

    def solve(self, problem: List[Tuple[np.ndarray, np.ndarray, int, int]]) -> List[np.ndarray]:
        results = []
        for h, x, up, down in problem:
            # Ensure NumPy arrays
            h_np = np.asarray(h, dtype=np.float64)
            x_np = np.asarray(x, dtype=np.float64)

            # Up‑sample x
            up_x = self._upsample(x_np, up)

            # Convolve with h (full)
            conv = np.convolve(up_x, h_np)

            # Down‑sample the result
            down_y = self._downsample(conv, down)

            results.append(down_y)
        return results