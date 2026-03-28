from typing import Any
import numpy as np
from scipy import signal


class Solver:
    def solve(self, problem: tuple) -> np.ndarray:
        a, b = problem
        mode = self.mode
        boundary = self.boundary
        result = signal.correlate2d(a, b, mode=mode, boundary=boundary)
        return result