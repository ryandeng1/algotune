import numpy as np

class Solver:
    def solve(self, problem: tuple) -> np.ndarray:
        a, b = problem
        return np.convolve(a, b, mode='full')
