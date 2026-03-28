from typing import Any
import numpy as np
from scipy import signal


class Solver:
    def solve(self, problem: tuple) -> np.ndarray:
        a, b = problem
        return signal.correlate2d(a, b, mode='full', boundary='fill')