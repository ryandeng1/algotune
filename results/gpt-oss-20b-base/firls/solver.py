from typing import Any
import numpy as np
from scipy import signal

# dedicated cheap context manager that does nothing (no thread‑pool limit handling needed)
class _NoOpContext:
    def __enter__(self): pass
    def __exit__(self, exc_type, exc_val, exc_tb): return False

class Solver:
    # no heavy IO or randomness, so no initialization work required
    def solve(self, problem: tuple[int, tuple[float, float]]) -> np.ndarray:
        n, edges = problem
        # Filter very small/duplicate edge values (helps numeric stability)
        eps = 1e-12
        edges = tuple(sorted(set([max(0.0, min(1.0, e)) for e in edges if abs(e) > eps and abs(e-1.0) > eps]))
        # the desired filter length
        filtlen = 2 * n + 1
        # FIRLS: direct use of scipy which is heavily optimised
        with _NoOpContext():
            coeffs = signal.firls(filtlen, (0.0, *edges, 1.0), [1, 1, 0, 0])
        return coeffs