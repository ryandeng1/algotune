import numpy as np

class Solver:
    def solve(self, problem: tuple) -> np.ndarray:
        a, b = problem
        n1, m1 = a.shape
        n2, m2 = b.shape
        out_shape = (n1 + n2 - 1, m1 + m2 - 1)
        a_padded = np.pad(a, ((0, n2 - 1), (0, m2 - 1)), 'constant', constant_values=0)
        b_padded = np.pad(b, ((0, n1 - 1), (0, m1 - 1)), 'constant', constant_values=0)
        A = np.fft.fft2(a_padded)
        B = np.fft.fft2(b_padded)
        C = A * B
        result = np.fft.ifft2(C)
        return np.real(result)
