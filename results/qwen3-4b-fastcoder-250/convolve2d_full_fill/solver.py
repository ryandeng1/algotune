import numpy as np
from numba import njit

class Solver:
    def solve(self, problem: tuple) -> np.ndarray:
        a, b = problem
        pad_h = b.shape[0] - 1
        pad_w = b.shape[1] - 1
        a_padded = np.pad(a, ((pad_h, 0), (pad_w, 0)), 'constant', constant_values=0)
        out_h = a_padded.shape[0]
        out_w = a_padded.shape[1]
        
        @njit
        def convolve(a_padded, b):
            result = np.zeros((out_h, out_w), dtype=np.float64)
            for i in range(out_h):
                for j in range(out_w):
                    total = 0.0
                    for ii in range(b.shape[0]):
                        for jj in range(b.shape[1]):
                            total += a_padded[i + ii, j + jj] * b[ii, jj]
                    result[i, j] = total
            return result
        
        return convolve(a_padded, b)
