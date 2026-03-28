from typing import Any
import numpy as np
from scipy import signal

class Solver:
    def solve(self, problem: tuple) -> np.ndarray:
        a, b = problem
        return signal.convolve2d(a, b)