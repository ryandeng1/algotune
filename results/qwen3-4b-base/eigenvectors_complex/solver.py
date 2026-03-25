import numpy as np

class Solver:
    def solve(self, problem: np.ndarray) -> list[list[complex]]:
        eigenvalues, eigenvectors = np.linalg.eig(problem)
        pairs = list(zip(eigenvalues, eigenvectors.T))
        pairs.sort(key=lambda pair: (-pair[0].real, -pair[0].imag))
        return [list(vec) for _, vec in pairs]
