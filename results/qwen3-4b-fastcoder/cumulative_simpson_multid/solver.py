from typing import Any
from numpy.typing import NDArray
from scipy.integrate import cumulative_simpson
import numpy as np

class Solver:
    def solve(self, problem: dict) -> NDArray:
        y2 = problem["y2"]
        dx = problem["dx"]
        y2 = np.asarray(y2)
        result = cumulative_simpson(y2, dx=dx)
        return result