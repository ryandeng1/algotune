from typing import Any
import numpy as np

class Solver:
    def __init__(self):
        self.mode = 'full'

    def solve(self, problem: tuple) -> np.ndarray:
        a, b = problem
        # Ensure inputs are numpy arrays for fast vectorized execution
        a = np.asarray(a)
        b = np.asarray(b)
        # numpy's convolve is highly optimized and avoids scipy overhead
        return np.convolve(a, b, mode=self.mode)