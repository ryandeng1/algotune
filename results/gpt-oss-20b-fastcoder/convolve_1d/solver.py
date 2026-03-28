from typing import Any
import numpy as np

class Solver:

    def __init__(self):
        self.mode = 'full'

    def solve(self, problem: tuple) -> np.ndarray:
        # Use NumPy's vectorised convolution which is typically faster than SciPy's
        a, b = problem
        return np.convolve(a, b, mode=self.mode)