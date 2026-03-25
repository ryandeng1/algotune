import numpy as np
from numba import njit

class Solver:
    def __init__(self, mode='full'):
        self.mode = mode

    def solve(self, problem):
        @njit
        def process_pair(a, b, mode):
            if mode == 'valid' and b.shape[0] > a.shape[0]:
                return np.empty(0)
            return np.correlate(a, b, mode=mode)
        
        results = []
        for a, b in problem:
            res = process_pair(a, b, self.mode)
            if res.size > 0:
                results.append(res)
        return results
