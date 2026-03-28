from typing import Any
import numpy as np
from scipy.fft import dctn

class Solver:
    def solve(self, problem: np.ndarray) -> np.ndarray:
        return dctn(problem, type=1)