import numpy as np

class Solver:
    def solve(self, problem: np.ndarray) -> np.ndarray:
        return np.fft.fftn(problem)
