import numpy as np

class Solver:
    def solve(self, problem: tuple) -> np.ndarray:
        a, b = problem
        b_flipped = np.flipud(np.fliplr(b))
        return np.convolve2d(a, b_flipped, mode='full')
