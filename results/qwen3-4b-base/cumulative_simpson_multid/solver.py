import numpy as np
from typing import Any
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: dict) -> NDArray:
        y2 = problem["y2"]
        dx = problem["dx"]
        axis = -1
        m = y2.shape[axis]
        if m < 2:
            return np.zeros_like(y2)
        weights = np.zeros(m)
        weights[0] = 1
        weights[-1] = 1
        for i in range(1, m - 1):
            weights[i] = 4 if i % 2 == 1 else 2
        weighted = y2 * weights
        return np.cumsum(weighted, axis=axis) * dx / 3