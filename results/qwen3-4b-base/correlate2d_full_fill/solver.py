import numpy as np

class Solver:
    def solve(self, problem: tuple) -> np.ndarray:
        a, b = problem
        b_flipped = np.flip(b, axis=0)
        b_flipped = np.flip(b_flipped, axis=1)
        
        m, n = a.shape
        p, q = b_flipped.shape
        output_size = (m + p - 1, n + q - 1)
        
        a_padded = np.zeros(output_size, dtype=a.dtype)
        a_padded[:m, :n] = a
        b_padded = np.zeros(output_size, dtype=b_flipped.dtype)
        b_padded[:p, :q] = b_flipped
        
        A = np.fft.fft2(a_padded)
        B = np.fft.fft2(b_padded)
        C = A * B
        result = np.fft.ifft2(C)
        return np.real(result)
