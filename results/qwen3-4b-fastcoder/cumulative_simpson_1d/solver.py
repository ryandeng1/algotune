from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: dict) -> np.ndarray:
        y = np.asarray(problem["y"])
        dx = problem["dx"]
        n = len(y)
        if n == 0:
            return np.array([])
        weights = np.ones(n)
        weights[0] = 1
        weights[-1] = 1
        for i in range(1, n - 1):
            weights[i] = 4 if i % 2 == 1 else 2
        return np.cumsum(y * weights) * dx / 3.0