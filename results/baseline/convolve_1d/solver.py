from typing import Any
import numpy as np
from scipy import signal

class Solver:

    def __init__(self):
        self.mode = 'full'

    def solve(self, problem: tuple) -> np.ndarray:
        a, b = problem
        return signal.convolve(a, b, mode=self.mode)
