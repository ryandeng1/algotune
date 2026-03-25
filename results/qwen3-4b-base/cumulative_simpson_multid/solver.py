import numpy as np

class Solver:
    def solve(self, problem: dict) -> np.ndarray:
        y2 = problem["y2"]
        dx = problem["dx"]
        N = y2.shape[-1]
        
        weights = np.ones(N)
        weights[1::2] = 4
        weights[2::2] = 2
        weights[-1] = 1
        
        return np.cumsum(y2 * weights, axis=-1) * dx / 3.0
