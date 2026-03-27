import numpy
from typing import Any
from scipy.fft import dctn
from numpy.typing import NDArray


class Solver:
    def solve(self, problem: NDArray) -> NDArray:
        """
        Compute the N‑dimensional DCT Type I.
        Uses the faster SciPy FFT interface.
        """
        # dctn works on any ND array and is implemented in C
        return dctn(problem, type=1)