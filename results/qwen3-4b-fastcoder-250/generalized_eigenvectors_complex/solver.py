import numpy as np
from scipy.linalg import eig as la_eig

class Solver:
    def solve(self, problem: tuple[np.ndarray, np.ndarray]) -> tuple[list[complex], list[list[complex]]]:
        A, B = problem
        n = A.shape[0]
        
        scale_B = np.sqrt(np.linalg.norm(B))
        B_scaled = B / scale_B
        A_scaled = A / scale_B
        
        eigenvalues, eigenvectors = la_eig(A_scaled, B_scaled)
        
        norms = np.linalg.norm(eigenvectors, axis=0)
        eigenvectors = eigenvectors / norms[:, np.newaxis]
        
        key = np.column_stack((-eigenvalues.real, -eigenvalues.imag))
        sorted_indices = np.argsort(key, axis=0)
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        
        sorted_eigenvalues = sorted_eigenvalues.tolist()
        sorted_eigenvectors = [list(vec) for vec in sorted_eigenvectors.T]
        
        return (sorted_eigenvalues, sorted_eigenvectors)
