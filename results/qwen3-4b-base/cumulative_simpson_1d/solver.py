import numpy as np

class Solver:
    def solve(self, problem: dict) -> np.ndarray:
        y = problem["y"]
        dx = problem["dx"]
        n = len(y)
        if n < 3:
            return np.zeros(n)
        weights = np.ones(n)
        weights[1:-1:2] = 4
        weights[2:-1:2] = 2
        return (dx / 3) * np.dot(weights, y)