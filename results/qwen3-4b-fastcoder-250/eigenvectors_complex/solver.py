import numpy as np

class Solver:
    def solve(self, problem: np.ndarray) -> list[list[complex]]:
        eigenvalues, eigenvectors = np.linalg.eig(problem)
        real_parts = np.real(eigenvalues)
        imag_parts = np.imag(eigenvalues)
        key = np.column_stack((-real_parts, -imag_parts))
        indices = np.argsort(key, axis=0)
        sorted_eigenvectors = eigenvectors[:, indices]
        return [vec.tolist() for vec in sorted_eigenvectors.T]
