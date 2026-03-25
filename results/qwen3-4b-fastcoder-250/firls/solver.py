import numpy as np

class Solver:
    def solve(self, problem):
        n, edges = problem
        N = 2 * n + 1
        edges = tuple(edges)
        M = 1000
        freqs = np.linspace(0, 1, M)
        A = np.exp(-2j * np.pi * np.arange(N)[:, None] * freqs[None, :])
        b = np.where(freqs < edges[1], 1.0, 0.0)
        return np.linalg.lstsq(A, b, rcond=None)[0]
