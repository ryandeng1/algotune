import numpy as np
from scipy import signal
from contextlib import nullcontext

def _single_thread_blas():
    return nullcontext()

class Solver:
    def solve(self, problem: tuple[int, tuple[float, float]]) -> np.ndarray:
        n, edges = problem
        # actual filter length (must be odd)
        n = 2 * n + 1
        # ensure edges is a tuple
        edges = tuple(edges)
        with _single_thread_blas():
            coeffs = signal.firls(n, (0.0, *edges, 1.0), [1, 1, 0, 0])
        return coeffs