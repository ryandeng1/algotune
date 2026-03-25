import numpy as np

class Solver:
    def __init__(self, mode='full'):
        self.mode = mode

    def solve(self, problem: tuple) -> np.ndarray:
        a, b = problem
        return np.convolve(a, b, mode=self.mode)
