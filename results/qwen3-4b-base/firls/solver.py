from typing import Any
from contextlib import nullcontext
import numpy as np
from scipy import signal


def _single_thread_blas():
    if threadpool_limits is None:
        return nullcontext()
    return threadpool_limits(limits=1)


class Solver:
    def solve(self, problem: tuple[int, tuple[float, float]]) -> np.ndarray:
        n, edges = problem
        n = 2 * n + 1

        freq = (0.0,) + edges + (1.0,)

        with _single_thread_blas():
            coeffs = signal.firls(n, freq, [1, 1, 0, 0])
        return coeffs