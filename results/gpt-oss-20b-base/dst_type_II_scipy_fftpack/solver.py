from typing import Any
import numpy as np
from numpy.typing import NDArray
from scipy import fft

class Solver:
    def solve(self, problem: NDArray) -> NDArray:
        """
        Compute the N‑dimensional DST Type II using the fast FFT backend.
        """
        # In SciPy's newer FFT module DST supports the optional 'type' argument.
        # For Type II we can call it directly. The function is vectorised and
        # performs well for large arrays.
        return fft.dst(problem, type=2)