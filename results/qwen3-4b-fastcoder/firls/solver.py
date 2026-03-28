from typing import Any
from contextlib import nullcontext
import numpy as np
from scipy import signal
import threadpool as threadpool_limits

def _single_thread_blas():
    if threadpool_limits is None:
        return nullcontext()
    return threadpool_limits(limits=1)

class Solver:
    def solve(self, problem: tuple[int, tuple[float, float]]) -> np.ndarray:
        n, edges = problem
        n = 2 * n + 1
        edges = tuple(edges)
        edge_freqs = (0.0, *edges, 1.0)
        with _single_thread_blas():
            return signal.firls(n, edge_freqs, [1, 1, 0, 0])