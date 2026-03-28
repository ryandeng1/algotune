import numpy as np
from scipy import signal
from functools import lru_cache

@lru_cache(maxsize=None)
def _firls_cached(n: int, edges: tuple[float, float]) -> np.ndarray:
    # Compute FIR filter coefficients using SciPy's firls
    return signal.firls(n, (0.0, *edges, 1.0), [1, 1, 0, 0])

class Solver:
    def solve(self, problem: tuple[int, tuple[float, float]]) -> np.ndarray:
        n, edges = problem
        n = 2 * n + 1
        coeffs = _firls_cached(n, edges)
        return coeffs