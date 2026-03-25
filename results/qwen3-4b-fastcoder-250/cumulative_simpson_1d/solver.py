import numpy as np

class Solver:
    def solve(self, problem: dict) -> np.ndarray:
        y = problem["y"]
        dx = problem["dx"]
        n = len(y)
        
        if n == 1:
            return np.array([0.0])
        
        weights = np.zeros(n)
        weights[0] = 1
        weights[-1] = 1
        for i in range(1, n - 1):
            weights[i] = 4 if i % 2 == 1 else 2
        
        cum = np.cumsum(weights * y)
        result = (dx / 3) * cum
        result[0] = 0
        return result
