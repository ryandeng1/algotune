import numpy as np

class Solver:
    def solve(self, problem: tuple[int, tuple[float, float]]) -> np.ndarray:
        n_input, edges = problem
        n = 2 * n_input + 1
        edges = tuple(edges)
        
        num_bands = 3
        num_samples = 2 * num_bands * 2
        freqs = np.linspace(0, 1, num_samples)
        
        A = np.exp(-1j * 2 * np.pi * freqs[:, None] * np.arange(n))
        
        mask = (freqs < edges[0]) | (freqs >= edges[1])
        b = np.where(mask, 1.0, 0.0)
        
        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return coeffs
