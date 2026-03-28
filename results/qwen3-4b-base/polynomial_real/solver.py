from typing import Any
from contextlib import nullcontext
import numpy as np


def _single_thread_blas():
    if threadpool_limits is None:
        return nullcontext()
    return threadpool_limits(limits=1)


class Solver:
    def solve(self, problem: list[float]) -> list[float]:
        coefficients = problem
        with _single_thread_blas():
            computed_roots = np.roots(coefficients)
        computed_roots = np.real_if_close(computed_roots, tol=1e-3)
        computed_roots = np.sort(computed_roots)[::-1]
        return computed_roots.tolist()