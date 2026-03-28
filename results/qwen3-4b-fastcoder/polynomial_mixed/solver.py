from typing import Any
from contextlib import nullcontext
import numpy as np


def _single_thread_blas():
    if threadpool_limits is None:
        return nullcontext()
    return threadpool_limits(limits=1)


class Solver:
    def solve(self, problem: list[float]) -> list[complex]:
        with _single_thread_blas():
            roots = np.roots(problem)
        sorted_roots = np.sort(roots)[::-1].tolist()
        return sorted_roots