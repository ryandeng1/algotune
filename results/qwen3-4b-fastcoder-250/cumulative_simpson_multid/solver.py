import numpy as np

class Solver:
    def solve(self, problem: dict) -> np.ndarray:
        y2 = problem["y2"]
        dx = problem["dx"]
        N = y2.shape[2]
        weights = np.zeros(N, dtype=np.float64)
        weights[0] = 1.0
        weights[1] = 4.0
        for i in range(2, N-1):
            weights[i] = 2.0 if i % 2 == 0 else 4.0
        weights[-1] = 1.0
        y2_reshaped = y2.reshape(-1, N)
        weighted = y2_reshaped * weights[:, np.newaxis]
        cumsum = np.cumsum(weighted, axis=1)
        result = (dx / 3.0) * cumsum.reshape(y2.shape)
        return result
