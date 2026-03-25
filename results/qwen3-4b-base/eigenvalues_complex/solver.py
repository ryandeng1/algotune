import numpy as np

class Solver:
    def solve(self, problem: np.ndarray) -> list[complex]:
        eigenvalues = np.linalg.eig(problem)[0]
        key = np.column_stack((-eigenvalues.real, -eigenvalues.imag))
        sorted_indices = np.argsort(key)
        return eigenvalues[sorted_indices].tolist()
