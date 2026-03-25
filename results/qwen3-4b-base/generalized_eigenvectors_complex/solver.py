import numpy as np
import scipy.linalg as la

class Solver:
    def solve(self, problem: tuple[np.ndarray, np.ndarray]) -> tuple[list[complex], list[list[complex]]]:
        A, B = problem
        n = A.shape[0]
        
        scale_B = np.sqrt(np.linalg.norm(B))
        B_scaled = B / scale_B
        A_scaled = A / scale_B
        
        eigenvalues, eigenvectors = la.eig(A_scaled, B_scaled)
        
        norms = np.linalg.norm(eigenvectors, axis=0)
        mask = norms > 1e-15
        if mask.any():
            eigenvectors[:, mask] = eigenvectors[:, mask] / norms[mask]
        
        key = np.column_stack([-np.real(eigenvalues), -np.imag(eigenvalues)])
        order = np.argsort(key, axis=0)
        sorted_eigenvalues = eigenvalues[order]
        sorted_eigenvectors = eigenvectors[:, order]
        
        eigenvalues_list = sorted_eigenvalues.tolist()
        eigenvectors_list = [list(vec.tolist()) for vec in sorted_eigenvectors]
        
        return (eigenvalues_list, eigenvectors_list)
