from typing import Any
from contextlib import nullcontext
import numpy as np


def _single_thread_blas():
    if threadpool_limits is None:
        return nullcontext()
    return threadpool_limits(limits=1)


class Solver:
    def solve(self, problem: list[float]) -> list[complex]:
        """
        Solve the polynomial problem by finding all roots of the polynomial.

        The polynomial is given as a list of coefficients [a_n, a_{n-1}, ..., a_0],
        representing p(x) = a_n * x^n + a_{n-1} * x^{n-1} + ... + a_0.
        The coefficients are real numbers.
        This method computes all the roots (which may be real or complex) and returns them
        sorted in descending order by their real parts and, if necessary, by their imaginary parts.

        :param problem: A list of polynomial coefficients (real numbers) in descending order.
        :return: A list of roots (real and complex) sorted in descending order.
        """
        coefficients = problem
        with _single_thread_blas():
            computed_roots = np.roots(coefficients)
        sorted_roots = sorted(computed_roots, key=lambda z: (z.real, z.imag), reverse=True)
        return sorted_roots
