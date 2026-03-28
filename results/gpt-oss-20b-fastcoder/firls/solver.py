from typing import Any
import numpy as np
from scipy import signal
from functools import lru_cache

class Solver:
    @lru_cache(maxsize=256)
    def _compute_coeffs(self, n: int, edges: tuple[float, ...]) -> np.ndarray:
        # Compute FIR filter coefficients using SciPy's firls
        return signal.firls(n, (0.0, *edges, 1.0), [1, 1, 0, 0])

    def solve(self, problem: tuple[int, tuple[float, float]]) -> np.ndarray:
        n, edges = problem
        # Problem definition: 2 * n + 1 filter taps
        taps = 2 * n + 1
        # edges is already a tuple of float values
        return self._compute_coeffs(taps, edges)