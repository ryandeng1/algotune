import numpy as np

class Solver:
    def __init__(self, mode='full'):
        self.mode = mode

    def solve(self, problem: list) -> list:
        return [
            np.convolve(a, b[::-1], mode=self.mode)
            for a, b in problem
            if not (self.mode == "valid" and b.shape[0] > a.shape[0])
        ]
